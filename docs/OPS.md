
## Git remotes
- origin (GitHub) = canonical
- gitlab remote kept but excluded from default fetch:
  git config remote.gitlab.skipDefaultUpdate true
Reason: SSH key not configured on GitLab; prevents `git fetch --all --tags` failures.
