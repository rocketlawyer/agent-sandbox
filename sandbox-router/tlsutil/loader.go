// Copyright 2026 The Kubernetes Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package tlsutil provides hot-reloading server certificate management and
// helpers for assembling tls.Config values from sandbox-router configuration.
package tlsutil

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/go-logr/logr"
)

// ReloadCallback is invoked after every reload attempt. ok reports whether
// the reload succeeded; err carries the parse/IO error on failure. It is
// optional and primarily exists so metrics can be wired without coupling
// this package to a particular metrics library.
type ReloadCallback func(ok bool, err error)

// CertReloader keeps an on-disk certificate pair, server or client, fresh in
// memory. The current *tls.Certificate is stored in an atomic.Pointer so
// handshake callbacks read it without locks; swaps only happen on a
// successful parse so we never serve a half-written file.
type CertReloader struct {
	certFile string
	keyFile  string
	log      logr.Logger
	cb       ReloadCallback

	cur atomic.Pointer[tls.Certificate]
}

// NewCertReloader loads the initial certificate pair and returns a reloader
// ready to be wired into tls.Config.GetCertificate or GetClientCertificate.
// certFile and keyFile may name the same file, holding both key and chain.
// cb may be nil.
func NewCertReloader(certFile, keyFile string, log logr.Logger, cb ReloadCallback) (*CertReloader, error) {
	if certFile == "" || keyFile == "" {
		return nil, errors.New("certFile and keyFile must be non-empty")
	}
	r := &CertReloader{certFile: certFile, keyFile: keyFile, log: log, cb: cb}
	if err := r.reload(); err != nil {
		return nil, fmt.Errorf("initial cert load: %w", err)
	}
	return r, nil
}

// GetCertificate implements crypto/tls.Config.GetCertificate. It returns the
// most recently loaded certificate. The returned pointer is stable for the
// duration of one handshake.
func (r *CertReloader) GetCertificate(_ *tls.ClientHelloInfo) (*tls.Certificate, error) {
	if c := r.cur.Load(); c != nil {
		return c, nil
	}
	return nil, errors.New("no certificate loaded")
}

// GetClientCertificate implements crypto/tls.Config.GetClientCertificate so
// the same reloader can supply the router's client certificate on upstream
// connections.
func (r *CertReloader) GetClientCertificate(_ *tls.CertificateRequestInfo) (*tls.Certificate, error) {
	return r.GetCertificate(nil)
}

// reload reads the cert/key pair from disk, parses it, and atomically swaps
// it into r.cur on success. Failures leave the previous certificate in place.
func (r *CertReloader) reload() error {
	cert, err := r.load()
	if err != nil {
		if r.cb != nil {
			r.cb(false, err)
		}
		return err
	}
	r.cur.Store(&cert)
	if r.cb != nil {
		r.cb(true, nil)
	}
	return nil
}

// load parses the pair. tls.LoadX509KeyPair reads the two files separately,
// so when they are one file, as with a Pod Certificates credential bundle, it
// is read here once instead: key and chain then always come from the same
// version of the file.
func (r *CertReloader) load() (tls.Certificate, error) {
	if r.certFile != r.keyFile {
		return tls.LoadX509KeyPair(r.certFile, r.keyFile)
	}
	data, err := os.ReadFile(r.certFile)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.X509KeyPair(data, data)
}

// Start watches the directories holding cert and key and reloads the pair,
// debounced, after changes to them (see fileWatcher). The goroutine exits
// when ctx is canceled.
func (r *CertReloader) Start(ctx context.Context) error {
	return (&fileWatcher{
		files:  []string{r.certFile, r.keyFile},
		what:   "cert",
		log:    r.log,
		reload: r.reload,
	}).start(ctx)
}

// fileWatcher calls reload, debounced, after events on one of files or on a
// Kubernetes AtomicWriter "..*" entry in their directories. A failed reload
// is logged and not retried until the next such event. what names the files
// in log messages.
type fileWatcher struct {
	files  []string
	what   string
	log    logr.Logger
	reload func() error
}

// start watches the directories holding w.files and runs the watcher
// goroutine until ctx is canceled.
//
// We watch the parent directories rather than the files themselves because
// the typical update pattern (cert-manager, Kubernetes Secret projection)
// renames a temporary file over the target, which detaches a file-level
// watch on most filesystems. Directory watches survive these renames.
func (w *fileWatcher) start(ctx context.Context) error {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("fsnotify watcher: %w", err)
	}

	dirs := map[string]struct{}{}
	for _, f := range w.files {
		dirs[filepath.Dir(f)] = struct{}{}
	}
	for d := range dirs {
		if err := watcher.Add(d); err != nil {
			_ = watcher.Close()
			return fmt.Errorf("watch %s: %w", d, err)
		}
	}

	go w.run(ctx, watcher)
	return nil
}

// run is the watcher goroutine. It coalesces bursts of events with a short
// debounce so a multi-file rotation triggers a single reload rather than
// several racy ones.
func (w *fileWatcher) run(ctx context.Context, watcher *fsnotify.Watcher) {
	defer func() {
		_ = watcher.Close()
	}()

	const debounce = 250 * time.Millisecond
	var (
		timer     *time.Timer
		timerCh   <-chan time.Time
		stopTimer = func() {
			if timer != nil && !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer = nil
			timerCh = nil
		}
	)
	defer stopTimer()

	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-watcher.Events:
			if !ok {
				return
			}
			// Filter to events that plausibly affect our files. Two
			// shapes need to pass:
			//
			//   1. Direct edits — fsnotify reports the leaf file path.
			//      Same as the original behavior.
			//   2. K8s projected Secret rotation — kubelet's AtomicWriter
			//      writes to a fresh "..TIMESTAMP" directory, then
			//      atomically swaps a "..data" symlink at the parent.
			//      Neither event names our leaf file; the leaf is a
			//      symlink chain that resolves through "..data". So we
			//      additionally let any event on a "..*" sibling
			//      through, which is the AtomicWriter signature.
			//
			// Random unrelated files in the same directory still get
			// filtered out. The reload itself is the final guard — it
			// re-parses the file and swaps only on success, so a
			// spurious wake-up is at worst a wasted ReadFile.
			name := filepath.Clean(ev.Name)
			base := filepath.Base(name)
			affectsOurFile := strings.HasPrefix(base, "..") // K8s AtomicWriter prefix
			for _, f := range w.files {
				affectsOurFile = affectsOurFile || name == filepath.Clean(f)
			}
			if !affectsOurFile {
				continue
			}
			stopTimer()
			timer = time.NewTimer(debounce)
			timerCh = timer.C
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			w.log.Error(err, w.what+" watcher error")
		case <-timerCh:
			timerCh = nil
			timer = nil
			if err := w.reload(); err != nil {
				w.log.Error(err, w.what+" reload failed", "files", w.files)
			} else {
				w.log.Info(w.what+" reloaded", "files", w.files)
			}
		}
	}
}

// CAReloader keeps a CA bundle fresh in memory, like CertReloader does for a
// certificate pair: the pool is swapped atomically, and only when the file
// yields at least one certificate (see LoadCAPool), so an unreadable, empty
// or unparseable bundle leaves the previous one in use. A bundle cut off
// after its first complete certificates is accepted with the rest missing,
// so writers should replace the file atomically, as Kubernetes volume
// projection does.
type CAReloader struct {
	file string
	log  logr.Logger

	cur atomic.Pointer[x509.CertPool]
}

// NewCAReloader loads the initial bundle and returns a reloader whose Pool
// follows the file once Start is called.
func NewCAReloader(file string, log logr.Logger) (*CAReloader, error) {
	r := &CAReloader{file: file, log: log}
	if err := r.reload(); err != nil {
		return nil, fmt.Errorf("initial CA load: %w", err)
	}
	return r, nil
}

// Pool returns the most recently loaded bundle.
func (r *CAReloader) Pool() *x509.CertPool {
	return r.cur.Load()
}

// reload reads and parses the bundle and swaps it in on success.
func (r *CAReloader) reload() error {
	pool, err := LoadCAPool(r.file)
	if err != nil {
		return err
	}
	r.cur.Store(pool)
	return nil
}

// Start watches the bundle's directory and reloads the bundle, debounced,
// after changes to it (see fileWatcher). The goroutine exits when ctx is
// canceled.
func (r *CAReloader) Start(ctx context.Context) error {
	return (&fileWatcher{
		files:  []string{r.file},
		what:   "CA bundle",
		log:    r.log,
		reload: r.reload,
	}).start(ctx)
}
