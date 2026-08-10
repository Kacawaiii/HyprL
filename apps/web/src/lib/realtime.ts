/**
 * Transport abstraction for a future live feed.
 *
 * There is no live market stream, so there is no implementation here — only
 * the shape a later phase will fill. Shipping a polling loop or an idle
 * WebSocket now would mean maintaining a connection to nothing and inviting a
 * page to render data that does not exist.
 */

export type RealtimeState = 'disconnected' | 'connecting' | 'connected' | 'error';

export interface RealtimeTransport {
  readonly state: RealtimeState;
  connect(): void;
  disconnect(): void;
  subscribe(listener: (event: { channel: string; payload: unknown }) => void): () => void;
}

/** The honest implementation for a system with no live feed. */
export function createNullTransport(): RealtimeTransport {
  return {
    state: 'disconnected',
    connect() { /* no live source exists in this phase */ },
    disconnect() { /* nothing to tear down */ },
    subscribe() { return () => undefined; },
  };
}
