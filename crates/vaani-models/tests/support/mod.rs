//! Shared test support: a temporary directory tree and a real HTTP server.
//!
//! # Why a hand-rolled HTTP server
//!
//! The download path has to be tested against something that speaks HTTP, not
//! against a mock: the behaviour under test *is* HTTP behaviour — `Range`
//! headers, `206` versus `200`, `Content-Range`, a body that stops early, a
//! 302 to a CDN. A mocked client proves only that the mock agrees with itself.
//!
//! And `dev-dependencies` are not available: this crate's `Cargo.toml` is
//! frozen and lists none, so a test framework or a server library cannot be
//! added. `std::net::TcpListener` and a bit of `HTTP/1.1` is what is available,
//! and it is enough to be a real server on a real socket.
//!
//! # The routes
//!
//! | path | behaviour |
//! |---|---|
//! | `/file/<name>` | the named body; honours `Range` with `206` + `Content-Range` |
//! | `/norange/<name>` | the named body, always `200` and always whole, and **no** `Accept-Ranges` |
//! | `/slow/<name>` | the body in slow slices, so a cancellation has somewhere to land |
//! | `/flaky/<name>` | `503` for the first `n` requests, then `/file` behaviour |
//! | `/truncated/<name>` | advertises the full length, sends half, closes |
//! | `/garbage/<name>` | the right length, the wrong bytes |
//! | `/missing` | `404` |
//! | `/unauthorised` | `401` |
//! | `/forbidden` | `403` |
//! | `/redirect/<name>` | `302` to `/file/<name>`, the Hugging Face CDN shape |

#![allow(dead_code)] // each test binary uses a different subset

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Temporary directories
// ---------------------------------------------------------------------------

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A directory that deletes itself.
///
/// `tempfile` is not available (no dev-dependencies), so this is the same idea in
/// thirty lines: a name nobody else will pick, and a `Drop` that removes it.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// A fresh empty directory named after `tag`.
    pub fn new(tag: &str) -> TempDir {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!("vaani-a7-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("temp dir must be creatable");
        TempDir { path }
    }

    /// The directory itself.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A path inside it.
    pub fn join(&self, rel: &str) -> PathBuf {
        self.path.join(rel)
    }

    /// Create `rel` (and its parents) as a directory.
    pub fn dir(&self, rel: &str) -> PathBuf {
        let p = self.join(rel);
        std::fs::create_dir_all(&p).expect("subdirectory must be creatable");
        p
    }

    /// Write a file of exactly `len` bytes, deterministically.
    ///
    /// The content is a repeating ramp rather than zeroes so that a hash
    /// mismatch has more than one possible cause, and so a truncated file is not
    /// accidentally the same bytes as a shorter one.
    pub fn file(&self, rel: &str, len: usize) -> PathBuf {
        let p = self.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("parent must be creatable");
        }
        std::fs::write(&p, ramp(len)).expect("file must be writable");
        p
    }

    /// Write arbitrary bytes.
    pub fn bytes(&self, rel: &str, data: &[u8]) -> PathBuf {
        let p = self.join(rel);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).expect("parent must be creatable");
        }
        std::fs::write(&p, data).expect("file must be writable");
        p
    }

    /// The SHA-256 of a file, as lower-case hex.
    pub fn sha256_of(path: &Path) -> String {
        use sha2::{Digest, Sha256};
        let mut f = std::fs::File::open(path).expect("file must exist");
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = f.read(&mut buf).expect("readable");
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        hex::encode(h.finalize())
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// `len` deterministic bytes.
pub fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// `len` deterministic bytes that are *not* the ramp, for "right size, wrong
/// content".
pub fn other_ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 7 + 13) % 251) as u8).collect()
}

// ---------------------------------------------------------------------------
// The HTTP server
// ---------------------------------------------------------------------------

/// A body the server will serve.
#[derive(Debug, Clone)]
pub struct Body {
    /// The bytes, and the `Content-Length` advertised.
    pub data: Vec<u8>,
    /// How many bytes to actually send before closing. `None` sends all of
    /// them.
    pub send: Option<usize>,
    /// Whether to honour `Range`.
    pub honour_range: bool,
    /// Milliseconds to sleep between slices.
    pub slice_delay_ms: u64,
    /// How many times to fail with `503` before serving.
    pub fail_times: u32,
    /// Whether to answer `302` to the plain route first.
    pub redirect_first: bool,
}

impl Body {
    /// A plain body.
    pub fn new(data: Vec<u8>) -> Body {
        Body {
            data,
            send: None,
            honour_range: true,
            slice_delay_ms: 0,
            fail_times: 0,
            redirect_first: false,
        }
    }

    /// A body the server will serve whole, ignoring `Range`.
    pub fn without_range(data: Vec<u8>) -> Body {
        Body {
            honour_range: false,
            ..Body::new(data)
        }
    }

    /// A body that stops early while advertising its full length.
    pub fn truncated(data: Vec<u8>, send: usize) -> Body {
        Body {
            send: Some(send),
            ..Body::new(data)
        }
    }

    /// A body served in `ms`-delayed slices.
    pub fn slow(data: Vec<u8>, ms: u64) -> Body {
        Body {
            slice_delay_ms: ms,
            ..Body::new(data)
        }
    }

    /// A body that answers `503` `n` times first.
    pub fn flaky(data: Vec<u8>, n: u32) -> Body {
        Body {
            fail_times: n,
            ..Body::new(data)
        }
    }

    /// A body the server redirects to itself once, the way a Hub `resolve/` URL
    /// redirects to a CDN.
    pub fn redirected(data: Vec<u8>) -> Body {
        Body {
            redirect_first: true,
            ..Body::new(data)
        }
    }
}

/// A running HTTP server on `127.0.0.1`.
#[derive(Debug)]
pub struct TestServer {
    port: u16,
    hits: Arc<Mutex<Vec<String>>>,
    counts: Arc<Mutex<BTreeMap<String, u32>>>,
    shutdown: Arc<AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TestServer {
    /// Start a server serving `bodies`, keyed by the last path segment.
    ///
    /// The special names `missing`, `unauthorised` and `forbidden` are answered
    /// with the corresponding status and are not looked up in `bodies`.
    pub fn start(bodies: BTreeMap<String, Body>) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let port = listener.local_addr().expect("bound").port();
        listener.set_nonblocking(true).expect("non-blocking");
        let bodies = Arc::new(bodies);
        let hits = Arc::new(Mutex::new(Vec::new()));
        let counts = Arc::new(Mutex::new(BTreeMap::new()));
        let shutdown = Arc::new(AtomicUsize::new(0));

        let t_bodies = Arc::clone(&bodies);
        let t_hits = Arc::clone(&hits);
        let t_counts = Arc::clone(&counts);
        let t_shutdown = Arc::clone(&shutdown);
        let handle = std::thread::spawn(move || {
            while t_shutdown.load(Ordering::SeqCst) == 0 {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let bodies = Arc::clone(&t_bodies);
                        let hits = Arc::clone(&t_hits);
                        let counts = Arc::clone(&t_counts);
                        std::thread::spawn(move || {
                            let _ = serve(stream, &bodies, &hits, &counts);
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    // A connection that was reset before it was accepted, or an
                    // interrupted syscall, must NOT take the server down. An
                    // earlier version broke out of the loop on any error, which
                    // made every test after the first dropped connection fail
                    // with an opaque "error sending request".
                    Err(e) if is_transient_accept_error(&e) => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(e) => {
                        eprintln!("test server accept failed: {e}");
                        break;
                    }
                }
            }
        });

        TestServer {
            port,
            hits,
            counts,
            shutdown,
            handle: Some(handle),
        }
    }

    /// The port it is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// A URL for one served body.
    pub fn url(&self, name: &str) -> String {
        format!("http://127.0.0.1:{}/{name}", self.port)
    }

    /// How many requests have arrived, in order, as `"METHOD path"`.
    pub fn hits(&self) -> Vec<String> {
        self.hits.lock().map(|h| h.clone()).unwrap_or_default()
    }

    /// How many requests arrived for one path.
    pub fn hits_for(&self, needle: &str) -> usize {
        self.hits().iter().filter(|h| h.contains(needle)).count()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.shutdown.store(1, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Whether a failed `accept` is worth retrying rather than fatal.
fn is_transient_accept_error(e: &std::io::Error) -> bool {
    if e.kind() == std::io::ErrorKind::Interrupted {
        return true;
    }
    // 10054 WSAECONNABORTED, 10053 WSAECONNRESET, 10022 WSAEINVAL on a
    // non-blocking socket: a peer that vanished between connect and accept.
    matches!(e.raw_os_error(), Some(10054) | Some(10053) | Some(10022))
}

fn serve(
    mut stream: TcpStream,
    bodies: &BTreeMap<String, Body>,
    hits: &Mutex<Vec<String>>,
    counts: &Mutex<BTreeMap<String, u32>>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut range: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':')
            && k.eq_ignore_ascii_case("range")
        {
            range = Some(v.trim().to_string());
        }
    }
    let _ = method;
    if let Ok(mut h) = hits.lock() {
        h.push(format!("GET {path}"));
    }

    // `/file/<name>` and `/<name>` are the same object: the redirect target is
    // the "CDN" copy, which lives at a different path on the same server.
    let trimmed = path.trim_start_matches('/');
    let trimmed = trimmed.strip_prefix("file/").unwrap_or(trimmed);
    let name = trimmed.rsplit('/').next().unwrap_or("").to_string();
    match name.as_str() {
        "missing" => return status(&mut stream, 404, "Not Found", b"no such file"),
        "unauthorised" => return status(&mut stream, 401, "Unauthorized", b"token required"),
        "forbidden" => return status(&mut stream, 403, "Forbidden", b"not your repo"),
        _ => {}
    }

    let Some(body) = bodies.get(&name) else {
        return status(&mut stream, 404, "Not Found", b"no such body");
    };
    let body = body.clone();

    // Per-name request counter, so a "fails twice then succeeds" body is real
    // rather than a sleep.
    let seen = counts
        .lock()
        .map(|mut c| {
            let e = c.entry(name.clone()).or_insert(0);
            *e += 1;
            *e
        })
        .unwrap_or(1);
    if body.fail_times > 0 && seen <= body.fail_times {
        return status(&mut stream, 503, "Service Unavailable", b"try again");
    }

    if body.redirect_first && range.is_none() && seen == 1 {
        // Only the first request for this object redirects, so a fresh fetch is
        // a 302 followed by a 200 while a *resumed* fetch goes straight to the
        // CDN -- which is what a real CDN does and what makes the resume cheap.
        let head =
            format!("HTTP/1.1 302 Found\r\nLocation: /file/{name}\r\nContent-Length: 0\r\n\r\n");
        stream.write_all(head.as_bytes())?;
        return stream.flush();
    }

    let total = body.data.len() as u64;
    let mut start = 0u64;
    if let Some(r) = &range
        && body.honour_range
        && let Some(spec) = r.strip_prefix("bytes=")
        && let Some((a, _)) = spec.split_once('-')
    {
        start = a.trim().parse().unwrap_or(0);
    }
    if start >= total {
        return status(&mut stream, 416, "Range Not Satisfiable", b"past the end");
    }
    let range_honoured = body.honour_range && start > 0;

    let to_send = total.saturating_sub(start) as usize;
    let advertised = total;
    let will_send = body.send.unwrap_or(to_send).min(to_send);
    let body_bytes = &body.data[start as usize..start as usize + will_send];

    let head = if range_honoured {
        // A 206 advertises the length of *this* response, not the whole
        // resource. Getting this wrong is what makes a resumed download fail
        // with an opaque body-decoding error, which is exactly the kind of
        // subtle server bug a hand-rolled server exists to avoid reproducing.
        format!(
            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{end}/{advertised}\r\nContent-Length: {to_send}\r\n\r\n",
            end = start + to_send as u64 - 1
        )
    } else {
        format!("HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {advertised}\r\n\r\n")
    };
    stream.write_all(head.as_bytes())?;

    if body.slice_delay_ms == 0 {
        stream.write_all(body_bytes)?;
    } else {
        // Slices, so a cancellation has a real window to be noticed in. Without
        // this a 4 KB body arrives in one chunk and "cancellable mid-download"
        // would be a claim about a code path that never runs.
        let slice = 1024usize;
        for chunk in body_bytes.chunks(slice) {
            stream.write_all(chunk)?;
            stream.flush()?;
            std::thread::sleep(Duration::from_millis(body.slice_delay_ms));
        }
    }
    // Close cleanly, then wait a moment before the socket is dropped. Windows
    // sends an RST when a socket with unread inbound data is closed, and hyper
    // reports that as "connection closed before message completed".
    // `Shutdown::Both` sends the FIN properly and the pause gives the peer time
    // to drain; without both of these, roughly one request in fifty failed for
    // reasons that had nothing to do with the code under test.
    stream.flush()?;
    let _ = stream.shutdown(std::net::Shutdown::Both);
    std::thread::sleep(Duration::from_millis(5));
    Ok(())
}

fn status(stream: &mut TcpStream, code: u16, reason: &str, body: &[u8]) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}
