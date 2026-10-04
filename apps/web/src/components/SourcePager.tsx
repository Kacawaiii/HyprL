export function SourcePager({ page, previous, next }: {
  page: number; previous?: () => void; next?: () => void;
}) {
  if (!previous && !next) return null;
  return <nav aria-label="Source pages" className="kv">
    <button className="control" disabled={!previous} onClick={previous}>Previous page</button>
    <span>Page {page}</span>
    <button className="control" disabled={!next} onClick={next}>Next page</button>
  </nav>;
}
