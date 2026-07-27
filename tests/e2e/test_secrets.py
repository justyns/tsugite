"""Secrets view: a secret set via the API shows its name and never its value.

Needs `writable_secrets_backend` - the daemon's default "env" backend rejects
writes outright, so the write has to happen against a swapped-in file backend.
"""

from playwright.sync_api import expect

from .helpers import auth_headers, open_view


def test_secret_name_appears_value_never_in_dom(authenticated_page, base_url, e2e_auth_token, writable_secrets_backend):
    secret_value = "sup3r-s3cr3t-value"
    resp = authenticated_page.request.post(
        f"{base_url}/api/secrets/e2e-test-secret",
        headers=auth_headers(e2e_auth_token),
        data={"value": secret_value},
    )
    assert resp.ok, resp.text()

    page = authenticated_page
    open_view(page, "secrets")

    expect(page.get_by_text("e2e-test-secret")).to_be_visible()
    assert secret_value not in page.content()
