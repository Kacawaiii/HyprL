# Operator deployment: member v1

These steps are a handoff, not an executed deployment. No nginx, system unit,
other application, bot, trader timer or runtime tool may be edited by a coding
agent. Keep sales closed until the public site's legal prerequisites and the
acceptance below are complete. The member feature defaults to disabled.

1. Review the branch and pinned commit from the delivery report. Fetch only the
   member branch. Keep the assigned base and existing external subscriber store.
   Prepare a new source release with `git archive <reviewed-commit>
   apps/hyprl-site`, extracting with `--strip-components=2`. Preserve all public
   site publisher/mediator/privacy facts supplied by the operator since the
   imported compliance release. Do not overwrite live edits blindly.
2. Set `MEMBER_CHECKOUT`, `MEMBER_PYTHON`, `SITE_RELEASE`, `SITE_ENV_FILE`,
   `MEMBER_HANDOFF_FILE` and the three member runtime/socket variables to
   dedicated operator-owned locations outside every source release. Create a
   private Python venv and install the pinned root `requirements-member.txt`.
   Do not reuse or change trader environments. Install the site's lockfile with
   `npm --prefix "$SITE_RELEASE" ci --ignore-scripts`; run its `npm test`.
3. In Discord's developer portal, create/review the OAuth application and a
   client secret. Register exactly
   `https://hyprlcore.com/auth/discord/callback`. Request only `identify` and
   `guilds.members.read`; configure the verified guild and the five distinct
   role IDs. Store client ID/secret, role mapping and stable random identity
   key only in the private site environment. No bot token belongs in this env.
   No real Discord call is authorized by the build; resolve the host's messaging
   egress restriction with the operator before acceptance.
4. Create the dedicated snapshot directory (0750) and member-state parent
   directory (0700), and set the absolute paths. Do not point to any research,
   trader, archive, holdout, account or original store directory. The site opens
   only `member-v1.json` read-only. For OS-level separation use the operator's
   existing service sandbox/read-only mount if available; mode 0440 prevents
   write-through, but the same unprivileged owner can replace a file in a writable
   directory. This code alone cannot impose process-level OS isolation.
5. Connect an approved producer to the handoff contract. Publish a current,
   reviewed observed snapshot with the exporter command in README. Start with
   genuine empty arrays when no verified evidence exists. Never install test
   fixtures in the runtime snapshot directory. Schedule the exporter through
   the existing approved job runner; source acquisition has separate grants.
6. If alerts are activated, review the original alert-router staff acceptance
   runbook. Do not reuse a bot's token implicitly or modify an existing bot.
   Keep its encryption key/store/policy separate from Node. Add only approved
   verified `guild/user` customers to the private router policy. Set a separate
   random `MEMBER_ROUTER_KEY` in the site and bridge environments and the same
   `MEMBER_ROUTER_SOCKET`. The bridge's socket directory must already exist,
   mode 0700. Launch the following in an operator-owned foreground/tmux process:

   ```bash
   cd "$MEMBER_CHECKOUT"
   "$MEMBER_PYTHON" -m scripts.alert_router.member_bridge \
     --socket "$MEMBER_ROUTER_SOCKET"
   ```

   Set `HYPRL_ALERT_STATE`, `HYPRL_ALERT_KEY` and `HYPRL_ALERT_POLICY` exactly as
   in the approved router configuration. The bridge does not start the worker
   or send messages. Worker activation remains a separate staff acceptance.
   Shut down this process before removing its socket. Never unlink another
   process's socket to force a second bridge to start.
7. Keep `MEMBER_BILLING_ORIGIN` empty until a real provider is integrated. Keep
   SMTP's existing EU/DPA gate unchanged unless its independent review passed.
   No Stripe setting is required. There must be no paid sale while renewal and
   direct cancellation are unconnected.
8. Before enabling members, confirm the existing ingress suppresses/redacts OAuth
   callback query strings and request bodies, overwrites `X-Real-IP`, preserves
   CSP and HTTPS, and forwards all three namespaces `/membre`, `/auth/discord`,
   `/api/member/` to this site. Do not change nginx from the agent session. If
   any ingress change is required, stop for the operator to handle it.
9. Set `MEMBER_ENABLED=true` only in the new private staging environment, and
   run a staged loopback server on a free non-live port with a matching PUBLIC_URL
   and registered staging callback. Do not use test authentication in staging.
   Real OAuth must be explicitly permitted by host egress policy first. Never
   print environment values or capture real OAuth bodies/tokens as evidence.
10. Staff acceptance: authorize five controlled Discord members (or change one
    staff member's role sequentially); confirm all tiers/API boundaries, role
    removal, session idle/logout, separate watchlists and alert lists, source
    times, pending states, positive and negative forecast skill, proof receipt
    identity and recomputed digest, stale-export labeling, mobile/light/dark and
    keyboard use. Exercise approved real messaging only after the separate router
    egress decision. Check no callback code/secret appeared in logs. Store only
    identities/digests/counts as evidence, outside Git.
11. After acceptance, use the **existing site release procedure** from the public
    site report: `npm prune --omit=dev --ignore-scripts`, atomically switch only
    the reviewed code release, then restart only the existing user-owned site
    service in the operator session. External subscriber/member/router data and
    stable keys stay in place; no private-store copy or rollback is made. No
    system service, nginx or other user change is part of this branch.
12. Verify production `/` and `/membre/connexion` headers, an actual Discord login,
    tier denials, isolated data and logout. When rolling back, set members disabled
    or switch back to the prior code release only; never restore old personal data.
    Existing member sessions naturally expire on restart. Report the deployed
    commit and acceptance counts. Keep sales and customer alert delivery disabled
    until their real dependencies have been accepted.
