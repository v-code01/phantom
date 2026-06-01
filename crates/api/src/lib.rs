use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use scheduler::{Request as SchedulerRequest, Scheduler};

// ── Request / Response ────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct ServeRequest {
    pub tokens:   Vec<u32>,
    pub agent_id: u64,
    pub kv_data:  Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ServeResponse {
    pub cache_hit:  bool,
    pub block_ids:  Vec<u64>,
    pub latency_ns: u64,
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ApiError {
    OutOfBlocks,
    BadRequest(String),
    Internal,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ApiError::OutOfBlocks     => (StatusCode::INSUFFICIENT_STORAGE, "slab full").into_response(),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
            ApiError::Internal        => (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response(),
        }
    }
}

impl From<scheduler::SchedulerError> for ApiError {
    fn from(e: scheduler::SchedulerError) -> Self {
        use coherence::CoherenceError;
        use kv::CacheError;
        use scheduler::SchedulerError;
        match e {
            SchedulerError::Coherence(CoherenceError::KvError(CacheError::OutOfBlocks)) => {
                ApiError::OutOfBlocks
            }
            SchedulerError::Coherence(CoherenceError::KvError(CacheError::DataSizeMismatch)) => {
                ApiError::BadRequest("kv_data block size mismatch".into())
            }
            _ => ApiError::Internal,
        }
    }
}

// ── AppState ──────────────────────────────────────────────────────────────────

struct AppState<const B: usize> {
    scheduler: Arc<Scheduler<B>>,
    hits:      Arc<AtomicU64>,
    misses:    Arc<AtomicU64>,
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn serve_handler<const B: usize>(
    State(state): State<Arc<AppState<B>>>,
    Json(req): Json<ServeRequest>,
) -> Result<Json<ServeResponse>, ApiError> {
    let n_blocks = req.tokens.len() / B;
    if n_blocks == 0 || req.tokens.len() % B != 0 {
        return Err(ApiError::BadRequest(
            format!("tokens.len() must be a non-zero multiple of {B}")
        ));
    }
    if req.kv_data.is_empty() {
        return Err(ApiError::BadRequest("kv_data must not be empty".into()));
    }
    if req.kv_data.len() % n_blocks != 0 {
        return Err(ApiError::BadRequest(
            "kv_data.len() must be divisible by n_blocks".into()
        ));
    }

    let block_bytes = req.kv_data.len() / n_blocks;
    let kv_data: Vec<Vec<u8>> = req.kv_data.chunks(block_bytes).map(|c| c.to_vec()).collect();

    let sched_req = SchedulerRequest {
        tokens:  req.tokens,
        kv_data,
        agent:   req.agent_id as usize,
    };

    let scheduler = state.scheduler.clone();
    let result = tokio::task::spawn_blocking(move || {
        let t0 = std::time::Instant::now();
        let r = scheduler.handle(&sched_req);
        (r, t0.elapsed().as_nanos() as u64)
    })
    .await
    .map_err(|_| ApiError::Internal)?;
    let (result, latency_ns) = result;

    let r = result.map_err(ApiError::from)?;

    if r.cache_hit { state.hits.fetch_add(1, Ordering::Relaxed); }
    else           { state.misses.fetch_add(1, Ordering::Relaxed); }

    Ok(Json(ServeResponse {
        cache_hit:  r.cache_hit,
        block_ids:  r.blocks.iter().map(|id| id.0 as u64).collect(),
        latency_ns,
    }))
}

async fn metrics_handler<const B: usize>(
    State(state): State<Arc<AppState<B>>>,
) -> String {
    let hits   = state.hits.load(Ordering::Relaxed);
    let misses = state.misses.load(Ordering::Relaxed);
    let (used, total) = state.scheduler.stats();
    format!(
        "# HELP phantom_cache_hits_total Total cache hits\n\
         # TYPE phantom_cache_hits_total counter\n\
         phantom_cache_hits_total {hits}\n\
         # HELP phantom_cache_misses_total Total cache misses\n\
         # TYPE phantom_cache_misses_total counter\n\
         phantom_cache_misses_total {misses}\n\
         # HELP phantom_blocks_used Current allocated blocks\n\
         # TYPE phantom_blocks_used gauge\n\
         phantom_blocks_used {used}\n\
         # HELP phantom_blocks_total Total slab capacity\n\
         # TYPE phantom_blocks_total gauge\n\
         phantom_blocks_total {total}\n"
    )
}

fn build_router<const B: usize>(state: Arc<AppState<B>>) -> Router {
    Router::new()
        .route("/v1/serve", post(serve_handler::<B>))
        .route("/health",   get(|| async { "ok" }))
        .route("/metrics",  get(metrics_handler::<B>))
        .with_state(state)
}

/// Start the PHANTOM HTTP server on `addr`. Blocks until the server exits.
pub async fn serve<const B: usize>(
    scheduler: Scheduler<B>,
    addr: std::net::SocketAddr,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState::<B> {
        scheduler: Arc::new(scheduler),
        hits:   Arc::new(AtomicU64::new(0)),
        misses: Arc::new(AtomicU64::new(0)),
    });
    let app = build_router(state);
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use coherence::SyncEngine;
    use tower::ServiceExt;

    const TEST_B: usize = 16;
    const TEST_STRIDE: usize = 64;

    fn build_test_app(capacity: usize) -> Router {
        let engine    = SyncEngine::<TEST_B>::new_heap(capacity, TEST_STRIDE, 100);
        let scheduler = Scheduler::new(engine);
        let state = Arc::new(AppState::<TEST_B> {
            scheduler: Arc::new(scheduler),
            hits:   Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        });
        build_router(state)
    }

    fn serve_body(tokens: Vec<u32>, agent_id: u64, kv_data: Vec<u8>) -> Request<Body> {
        let json = serde_json::json!({
            "tokens":   tokens,
            "agent_id": agent_id,
            "kv_data":  kv_data,
        });
        Request::builder()
            .method("POST")
            .uri("/v1/serve")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&json).unwrap()))
            .unwrap()
    }

    async fn parse_serve_response(resp: axum::response::Response) -> ServeResponse {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = build_test_app(64);
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 64).await.unwrap();
        assert_eq!(&bytes[..], b"ok");
    }

    #[tokio::test]
    async fn cold_miss_returns_200_with_cache_hit_false() {
        let app = build_test_app(64);
        // B=16, 32 tokens = 2 blocks, kv_data = 2 * B * STRIDE = 2048 bytes
        let tokens:  Vec<u32> = (0u32..32).collect();
        let kv_data: Vec<u8>  = vec![1u8; 2 * TEST_B * TEST_STRIDE];
        let resp = app
            .oneshot(serve_body(tokens, 0, kv_data))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = parse_serve_response(resp).await;
        assert!(!body.cache_hit, "first request must be cold miss");
        assert_eq!(body.block_ids.len(), 2, "32 tokens / B=16 = 2 blocks");
    }

    #[tokio::test]
    async fn cache_hit_returns_true_on_second_request() {
        let app = build_test_app(64);
        let tokens:  Vec<u32> = (0u32..32).collect();
        let kv_data: Vec<u8>  = vec![1u8; 2 * TEST_B * TEST_STRIDE];

        let r1 = app.clone()
            .oneshot(serve_body(tokens.clone(), 0, kv_data.clone()))
            .await.unwrap();
        assert_eq!(r1.status(), StatusCode::OK);
        assert!(!parse_serve_response(r1).await.cache_hit);

        let r2 = app
            .oneshot(serve_body(tokens, 1, kv_data))
            .await.unwrap();
        assert_eq!(r2.status(), StatusCode::OK);
        assert!(parse_serve_response(r2).await.cache_hit, "second request must hit cache");
    }

    #[tokio::test]
    async fn slab_full_returns_507() {
        // capacity=2 holds exactly one 2-block (32-token) artifact
        let app = build_test_app(2);
        let kv_data: Vec<u8> = vec![1u8; 2 * TEST_B * TEST_STRIDE];

        let r1 = app.clone()
            .oneshot(serve_body((0u32..32).collect(), 0, kv_data.clone()))
            .await.unwrap();
        assert_eq!(r1.status(), StatusCode::OK, "first artifact must fit in slab");

        let r2 = app
            .oneshot(serve_body((100u32..132).collect(), 0, kv_data))
            .await.unwrap();
        assert_eq!(r2.status(), StatusCode::INSUFFICIENT_STORAGE, "full slab must return 507");
    }

    #[tokio::test]
    async fn bad_token_count_returns_400() {
        let app = build_test_app(64);
        // 31 tokens — not divisible by B=16
        let kv_data: Vec<u8> = vec![1u8; 2 * TEST_B * TEST_STRIDE];
        let resp = app
            .oneshot(serve_body((0u32..31).collect(), 0, kv_data))
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn latency_ns_is_positive() {
        let app = build_test_app(64);
        let tokens:  Vec<u32> = (0u32..32).collect();
        let kv_data: Vec<u8>  = vec![1u8; 2 * TEST_B * TEST_STRIDE];

        // Cold miss
        let r1 = app
            .oneshot(serve_body(tokens, 0, kv_data))
            .await.unwrap();
        let b1 = parse_serve_response(r1).await;
        assert!(b1.latency_ns > 0, "latency_ns must be non-zero");
    }

    #[tokio::test]
    async fn metrics_counts_hits_and_misses() {
        let app = build_test_app(64);
        let tokens:  Vec<u32> = (0u32..32).collect();
        let kv_data: Vec<u8>  = vec![1u8; 2 * TEST_B * TEST_STRIDE];

        app.clone().oneshot(serve_body(tokens.clone(), 0, kv_data.clone())).await.unwrap();
        app.clone().oneshot(serve_body(tokens, 1, kv_data)).await.unwrap();

        let resp = app
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body  = std::str::from_utf8(&bytes).unwrap();
        assert!(body.contains("phantom_cache_misses_total 1"), "body:\n{body}");
        assert!(body.contains("phantom_cache_hits_total 1"),   "body:\n{body}");
    }
}
