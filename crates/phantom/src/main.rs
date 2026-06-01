#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine    = coherence::SyncEngine::<16>::new_heap(1024, 64, 100);
    let scheduler = scheduler::Scheduler::new(engine);
    let addr: std::net::SocketAddr = "127.0.0.1:8080".parse()?;
    println!("PHANTOM M5a listening on {addr}");
    api::serve::<16>(scheduler, addr).await
}
