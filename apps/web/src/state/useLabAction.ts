/**
 * One Model Lab job control (build, launch, cancel, monitor) and its outcome.
 *
 * A control is sent once per click: no retry, no optimistic state. The server's answer is the
 * outcome; on success every cached `lab:` read is dropped so the job list shows the new state.
 */
import { useCallback, useRef, useState } from 'react';
import { ApiError } from '../api/client';
import { invalidate } from './useQuery';

export type ActionState<T> =
  | { status: 'idle' }
  | { status: 'sending' }
  | { status: 'done'; data: T }
  | { status: 'error'; message: string; code: number };

/** The sentence shown for a refused control: the server's reason, with what to do for the known cases. */
export function actionError(error: unknown): { message: string; code: number } {
  const code = error instanceof ApiError ? error.status : 0;
  const reason = error instanceof Error ? error.message : 'Unknown error';
  if (code === 401) return { code, message: 'The operator token was refused. Forget it and enter it again.' };
  if (code === 403) {
    return { code, message: 'The listener refused this page: controls are accepted only from a page served by the lab listener itself on a loopback address.' };
  }
  if (code === 503) return { code, message: 'Model Lab is not configured on this server.' };
  if (code === 0) return { code, message: `No answer from the server (${reason}). Nothing is known to have been queued.` };
  return { code, message: `Refused (${code}): ${reason}.` };
}

export function useLabAction<A extends unknown[], T>(send: (...args: A) => Promise<T>, onDone?: (data: T) => void) {
  const [state, setState] = useState<ActionState<T>>({ status: 'idle' });
  const busy = useRef(false);
  const sendRef = useRef(send);
  sendRef.current = send;
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  const run = useCallback(async (...args: A) => {
    if (busy.current) return;
    busy.current = true;
    setState({ status: 'sending' });
    try {
      const data = await sendRef.current(...args);
      invalidate('lab:');
      setState({ status: 'done', data });
      doneRef.current?.(data);
    } catch (error) {
      setState({ status: 'error', ...actionError(error) });
    } finally {
      busy.current = false;
    }
  }, []);
  const reset = useCallback(() => setState({ status: 'idle' }), []);
  return { state, run, reset };
}
