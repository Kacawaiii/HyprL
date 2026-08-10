/**
 * A windowed table.
 *
 * Signals, trades and events are long. Rendering 20 000 `<tr>` elements makes
 * the browser lay out 20 000 rows nobody is looking at, so only the visible
 * slice (plus a small overscan) is mounted. The scroll height is faked with
 * spacer rows so the scrollbar still reflects the real dataset.
 */
import { useRef, useState, type ReactNode } from 'react';

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  width?: string;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowHeight?: number;
  height?: number;
  overscan?: number;
  empty?: ReactNode;
}

export function DataTable<T>({
  rows, columns, rowHeight = 26, height = 420, overscan = 8, empty,
}: Props<T>) {
  const [scrollTop, setScrollTop] = useState(0);
  const viewport = useRef<HTMLDivElement>(null);

  if (rows.length === 0) {
    return <div className="state">{empty ?? 'No rows'}</div>;
  }

  const visible = Math.ceil(height / rowHeight);
  const first = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const last = Math.min(rows.length, first + visible + overscan * 2);
  const slice = rows.slice(first, last);

  return (
    <div
      className="table-scroll"
      ref={viewport}
      style={{ maxHeight: height }}
      onScroll={(event) => setScrollTop((event.target as HTMLDivElement).scrollTop)}
      data-testid="data-table"
      data-total-rows={rows.length}
      data-rendered-rows={slice.length}
    >
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} style={column.width ? { width: column.width } : undefined}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {first > 0 && <tr style={{ height: first * rowHeight }} aria-hidden="true" />}
          {slice.map((row, index) => (
            <tr key={first + index}>
              {columns.map((column) => (
                <td key={column.key}>{column.render(row)}</td>
              ))}
            </tr>
          ))}
          {last < rows.length && (
            <tr style={{ height: (rows.length - last) * rowHeight }} aria-hidden="true" />
          )}
        </tbody>
      </table>
    </div>
  );
}
