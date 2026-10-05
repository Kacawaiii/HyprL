/** The API docs page: it must link the reference and client, list only routes that exist in the
 *  generated OpenAPI document with the permission it declares, and never send a request. */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { App } from '../App';
import { API_BASE, DOC_LINKS, JOURNEY } from '../lib/apiDocs';

const spec = JSON.parse(readFileSync(resolve(__dirname, '../../../../docs/artifacts/b2b_openapi_v1.json'), 'utf8')) as {
  paths: Record<string, Record<string, Record<string, unknown>>>;
};

afterEach(() => vi.unstubAllGlobals());

describe('API docs page', () => {
  it('lists only routes of the generated OpenAPI document, with their permission', () => {
    for (const step of JOURNEY) {
      const operation = spec.paths[API_BASE + step.path]?.[step.method.toLowerCase()];
      expect(operation, `${step.method} ${step.path}`).toBeDefined();
      expect(operation?.['x-permission'] ?? operation?.['x-permissions']).toBe(step.permission);
    }
  });

  it('keeps the documented files in the repository', () => {
    for (const link of Object.values(DOC_LINKS)) {
      expect(() => readFileSync(resolve(__dirname, '../../../../', link.label))).not.toThrow();
      expect(link.href.endsWith(link.label)).toBe(true);
    }
  });

  it('renders the guide, example client and the main path, and sends no request', () => {
    const fetchMock = vi.fn(() => Promise.reject(new Error('no request expected')));
    vi.stubGlobal('fetch', fetchMock);
    render(<MemoryRouter initialEntries={['/api-docs']}><App /></MemoryRouter>);
    return screen.findByRole('table', { name: 'Main path' }).then((table) => {
      expect(screen.getByRole('link', { name: 'docs/API_B2B_V1.md' })).toHaveAttribute('href', DOC_LINKS.guide.href);
      expect(screen.getByRole('link', { name: 'examples/b2b_client.py' })).toHaveAttribute('href', DOC_LINKS.client.href);
      const rows = within(table).getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell')[0]?.textContent);
      expect(rows).toEqual(['Snapshot', 'Dataset', 'Model', 'Experiment', 'Prediction', 'Monitoring']);
      expect(screen.getByLabelText('Demo command')).toHaveTextContent('python -m examples.b2b_client --demo');
      expect(screen.getByText(/establish no market edge/)).toBeInTheDocument();
    });
  });
});
