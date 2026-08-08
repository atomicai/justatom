#!/usr/bin/env bash

list_compose_projects() {
  local containers volumes networks volume_projects network_projects volume network

  containers="$(docker ps -a --format '{{.Label "com.docker.compose.project"}}')" || return $?
  volumes="$(docker volume ls -q --filter label=com.docker.compose.project)" || return $?
  volume_projects="$(
    while IFS= read -r volume; do
      [[ -z "$volume" ]] && continue
      docker volume inspect --format '{{index .Labels "com.docker.compose.project"}}' "$volume" || exit $?
    done <<<"$volumes"
  )" || return $?
  networks="$(docker network ls -q --filter label=com.docker.compose.project)" || return $?
  network_projects="$(
    while IFS= read -r network; do
      [[ -z "$network" ]] && continue
      docker network inspect --format '{{index .Labels "com.docker.compose.project"}}' "$network" || exit $?
    done <<<"$networks"
  )" || return $?

  printf '%s\n' "$containers" "$volume_projects" "$network_projects" | sed '/^$/d' | sort -u
}

list_preexisting_compose_projects() {
  local projects

  projects="$(list_compose_projects)" || return $?
  awk -v project="$PROJECT" '$0 != project' <<<"$projects"
}

project_resources() {
  local containers volumes networks

  containers="$(docker ps -aq --filter "label=com.docker.compose.project=${PROJECT}")" || return $?
  volumes="$(docker volume ls -q --filter "label=com.docker.compose.project=${PROJECT}")" || return $?
  networks="$(docker network ls -q --filter "label=com.docker.compose.project=${PROJECT}")" || return $?

  [[ -z "$containers" ]] || printf '%s\n' "$containers"
  [[ -z "$volumes" ]] || printf '%s\n' "$volumes"
  [[ -z "$networks" ]] || printf '%s\n' "$networks"
  return 0
}
