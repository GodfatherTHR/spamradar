export default {
  async fetch(request, env) {
    const incoming = new URL(request.url);
    const origin = env.API_ORIGIN || "https://toriqulhaque-spamradar.static.hf.space";
    const target = new URL(incoming.pathname + incoming.search, origin);
    const headers = new Headers(request.headers);
    if (env.SPAMRADAR_API_KEY) {
      headers.set("X-API-Key", env.SPAMRADAR_API_KEY);
    }
    const response = await fetch(target.toString(), {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "follow",
    });
    const out = new Headers(response.headers);
    out.delete("content-encoding");
    out.delete("content-length");
    out.set("Access-Control-Allow-Origin", "*");
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: out,
    });
  },
};
