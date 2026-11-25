"""Response defenses; render untrusted text safely even when CSP is unavailable."""

# PyScript dynamically loads Python/WASM and external runtime resources. Do not
# silently break it with an unverified script-src/default-src allowlist.
PYSCRIPT_CSP = "base-uri 'self'; object-src 'none'; frame-ancestors 'none'; script-src-attr 'none'"
IMAGE_CSP = ("default-src 'none'; script-src 'self'; script-src-attr 'none'; "
             "style-src 'self'; img-src 'self' blob:; connect-src 'self'; "
             "base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'")


def security_headers(policy):
    return {'Content-Security-Policy': policy, 'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY', 'Referrer-Policy': 'strict-origin-when-cross-origin'}


class SecurityHeadersMiddleware:
    """Apply to static, JSON, redirect and handled error responses alike."""
    def __init__(self, app, policy):
        self.app, self.headers = app, security_headers(policy)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)

        async def secured_send(message):
            if message['type'] == 'http.response.start':
                from starlette.datastructures import MutableHeaders
                headers = MutableHeaders(scope=message)
                headers.update(self.headers)
            await send(message)

        await self.app(scope, receive, secured_send)
