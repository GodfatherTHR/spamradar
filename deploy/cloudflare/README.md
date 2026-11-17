# Cloudflare Worker proxy for SpamRadar

The live app is already deployed at:

https://toriqulhaque-spamradar.static.hf.space/

This Worker proxies the static app to your own subdomain. Inference still runs in the visitor's browser. The Worker only forwards files and adds no model compute.

## Option A: Cloudflare dashboard

1. Open Cloudflare Dashboard, then Workers and Pages, then Create, then Worker.
2. Paste the contents of `cloudflare-worker.js`.
3. Deploy the Worker. Give it a name such as `spamradar-proxy`.
4. In the Worker settings, add a variable named `API_ORIGIN` with this value:
   `https://toriqulhaque-spamradar.static.hf.space`
5. Add a custom domain or route for your real subdomain, for example `spam.yourdomain.com`.
6. Open the custom domain and confirm that the SpamRadar page loads and a prediction works.

## Option B: Wrangler

1. Replace `spam.yourdomain.com` and `yourdomain.com` in `wrangler.toml` with your real names.
2. Uncomment the `routes` block.
3. Export a Cloudflare API token with these permissions:
   - Account: Workers Scripts: Edit
   - Zone: Workers Routes: Edit
   - Zone: Zone: Read
4. Run:

```powershell
$env:CLOUDFLARE_API_TOKEN = "your-token"
npx wrangler deploy
```

If you omit the route, Wrangler gives you a free `workers.dev` URL. You can add the custom domain later in the dashboard.

## Free plan notes

The Worker free plan allows 100,000 requests per day. The static page loads a handful of files and the browser caches them. The 6.5 MB browser model is proxied once and can be cached by Cloudflare.

The actual custom subdomain cannot be attached from here because the real domain name and a Cloudflare API token or dashboard session were not provided. Replace the placeholders or send the domain and token if you want this step finished remotely.
