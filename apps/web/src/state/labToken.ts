/**
 * The operator token for the Model Lab read endpoints.
 *
 * Held in module memory only: never in the URL, localStorage, a cookie or a log, so a reload forgets it.
 * `version` changes on every set/clear and is part of each query key, so a cached answer obtained with
 * one token is never shown for another and clearing the token drops the data from view.
 */
import { useSyncExternalStore } from 'react';
import { invalidate } from './useQuery';

let token = '';
let version = 0;
const listeners = new Set<() => void>();

function emit(): void {
  version += 1;
  invalidate('lab:');
  listeners.forEach((listener) => listener());
}

export function setLabToken(value: string): void {
  token = value.trim();
  emit();
}

export function clearLabToken(): void {
  token = '';
  emit();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLabToken(): { token: string; version: number } {
  useSyncExternalStore(subscribe, () => version);
  return { token, version };
}
