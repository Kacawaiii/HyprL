/** API documentation for the B2B listener. Static: the cockpit never holds a project key and
 *  sends no B2B request; it points to the reference and the example client. */
import { API_BASE, DOC_LINKS, JOURNEY, LIMITS } from '../lib/apiDocs';

export function ApiDocsPage() {
  return (
    <div className="stack">
      <section className="card">
        <h2 className="card-title">B2B API v1</h2>
        <p>
          A separate authenticated listener exposes project resources under{' '}
          <code>{API_BASE}</code>. The cockpit is a different, read-only operator surface and is not
          a tenant endpoint.
        </p>
        <ul>
          <li><a href={DOC_LINKS.guide.href} target="_blank" rel="noreferrer">{DOC_LINKS.guide.label}</a>: guide, startup and permissions</li>
          <li><a href={DOC_LINKS.openapi.href} target="_blank" rel="noreferrer">{DOC_LINKS.openapi.label}</a>: generated OpenAPI 3.1</li>
          <li><a href={DOC_LINKS.client.href} target="_blank" rel="noreferrer">{DOC_LINKS.client.label}</a>: example client</li>
        </ul>
      </section>

      <section className="card">
        <h2 className="card-title">Main path</h2>
        <table className="data" aria-label="Main path">
          <thead><tr><th>Step</th><th>Request</th><th>Permission</th><th>Purpose</th></tr></thead>
          <tbody>
            {JOURNEY.map((row) => (
              <tr key={row.step}>
                <td>{row.step}</td>
                <td><code>{row.method} {row.path}</code></td>
                <td><code>{row.permission}</code></td>
                <td>{row.purpose}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <h2 className="card-title">Try it locally</h2>
        <pre aria-label="Demo command"><code>python -m examples.b2b_client --demo</code></pre>
        <p className="muted">
          Starts an ephemeral loopback listener with an in-memory key and runs the path above on
          labelled synthetic data. For a persistent server, keep the key in the environment
          (<code>HYPRL_B2B_KEY</code>); never put it in a URL, a file in Git or this page.
        </p>
      </section>

      <section className="card">
        <h2 className="card-title">Limits</h2>
        <ul>{LIMITS.map((text) => <li key={text}>{text}</li>)}</ul>
      </section>
    </div>
  );
}
