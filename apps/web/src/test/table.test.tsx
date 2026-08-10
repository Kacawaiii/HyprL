/** The table must not mount rows nobody can see. */
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { DataTable } from '../components/DataTable';

describe('virtualised table', () => {
  it('renders only a window of a large dataset', () => {
    const rows = Array.from({ length: 20_000 }, (_, index) => ({ id: index }));
    render(
      <DataTable
        rows={rows}
        height={400}
        rowHeight={26}
        columns={[{ key: 'id', header: 'ID', render: (row) => row.id }]}
      />,
    );
    const table = screen.getByTestId('data-table');
    expect(table.dataset.totalRows).toBe('20000');
    const rendered = Number(table.dataset.renderedRows);
    expect(rendered).toBeGreaterThan(0);
    // a screenful plus overscan, not twenty thousand DOM rows
    expect(rendered).toBeLessThan(60);
  });

  it('shows an empty state rather than an empty grid', () => {
    render(
      <DataTable rows={[]} columns={[{ key: 'a', header: 'A', render: () => null }]}
        empty="Nothing recorded" />,
    );
    expect(screen.getByText('Nothing recorded')).toBeInTheDocument();
  });
});
