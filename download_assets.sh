#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  bash download_assets.sh <asset-group> [<asset-group> ...]

Asset groups:
  ids            Glint360K ids_and_fns.npy
  fr-templates   Face-recognition Glint360K templates and norms
  clip           OpenCLIP Glint360K templates and norms
  templates      ids + fr-templates + clip
  sae            BatchTopK SAE weights
  dfd            Deep Face Decoder weights
  arc2face       Arc2Face, Stable Diffusion, and ArcFace recognizer archive
  all            templates + sae + dfd + arc2face
  list           Show files managed by this script

Before using this script, fill the URLS table with Dropbox share links whose
final query parameter is dl=1. The top-level Dropbox folder link is only for
browsing; wget needs a file-specific URL for each artifact or archive.
USAGE
}

die() {
  echo "download_assets.sh: $*" >&2
  exit 1
}

declare -A URLS=(
  # Glint360K index
  [ids_and_fns]="https://www.dropbox.com/scl/fi/tpkcrldtnf3qkysnvc7ar/ids_and_fns.npy?rlkey=2f5djtiwr7ukupdnzkw9agvfz&dl=1"

  # FR templates
  [swinface_norms]="https://www.dropbox.com/scl/fi/5bs2ohluzp7cbj5lpynsm/norms.pth?rlkey=aaedyrqvi8ptmpblz2n0qewub&dl=1"
  [swinface_templates]="https://www.dropbox.com/scl/fi/1tqvdqvq0yh1zn4fx638u/templates.pth?rlkey=jnnc61utb3nyrc9dxrar6ldy1&dl=1"
  [adaface_ires50_norms]="https://www.dropbox.com/scl/fi/xps5joiqbjx8qh7atxz0h/norms.pth?rlkey=anek36wgebo87m9qxs4h5jb2l&dl=1"
  [adaface_ires50_templates]="https://www.dropbox.com/scl/fi/zqmjlm8bn6b13af1icf7m/templates.pth?rlkey=rp2odorlvjjs74419gm2a0s9b&dl=1"
  [arcface_res100_norms]="https://www.dropbox.com/scl/fi/fljuf6dg980jgtjpq7qta/norms.pth?rlkey=4ev49ah24rlagjsgfrowur40f&dl=1"
  [arcface_res100_templates]="https://www.dropbox.com/scl/fi/unjztevkicnldc6ido4qm/templates.pth?rlkey=dqarjydr0pd1t330zsew14vfb&dl=1"
  [adaface_vitb_norms]="https://www.dropbox.com/scl/fi/kdkw0c49v87kplh5h9jwf/norms.pth?rlkey=upfnw1q6ct5gy6s0b7v17g1hu&dl=1"
  [adaface_vitb_templates]="https://www.dropbox.com/scl/fi/k1xkm3sny81ltx3hqln5r/templates.pth?rlkey=c94qdxcrz7fpc9r61u1y4p1ag&dl=1"

  # CLIP templates
  [clip_norms]="https://www.dropbox.com/scl/fi/zqzef3fm8zbw96u8odl5z/norms.pth?rlkey=2yyzi1dpggnhjd7tsglxnyf9h&dl=1"
  [clip_templates]="https://www.dropbox.com/scl/fi/7cmi1myc57lphpi8evair/templates_f16.pth?rlkey=xsfw6qnbst0f5yxh9j4zjuqc0&dl=1"

  # SAE weights
  [sae_swinface]="https://www.dropbox.com/scl/fi/gow0dbmdo6cpluj6jvmgt/weights.pt?rlkey=qop1x8q9pq5kztyl9a9sp1y14&dl=1"
  [sae_adaface_ires50]="https://www.dropbox.com/scl/fi/k10cbifub7klbl9sgc494/weights.pt?rlkey=9uz31w67asekqn3dbyvzdp5vu&dl=1"
  [sae_adaface_vitb]="https://www.dropbox.com/scl/fi/bfdf4lhk3d034z9atuna7/weights.pt?rlkey=sxt3z7muobtf4i0osgq46dfkl&dl=1"
  [sae_arcface_res100]="https://www.dropbox.com/scl/fi/sekd2x9wkggwhp8davzs3/weights.pt?rlkey=ach6pyrjpkhw76yq9zvii3p57&dl=1"

  # Deep Face Decoder weights
  [dfd_swinface]="https://www.dropbox.com/scl/fi/wl3e5u9yi5nlm7o22b4tn/best.pt?rlkey=afvvp3tmqkk0n72lnljekln8w&dl=1"
  [dfd_adaface_ires50]="https://www.dropbox.com/scl/fi/bbvwdatpi8e0fv4ypoofg/best.pt?rlkey=zlb406d3hx82ijp2akfjv8jmz&dl=1"
  [dfd_adaface_vitb]="https://www.dropbox.com/scl/fi/duw9m9wo71o9sycm0i4tt/best.pt?rlkey=1myqfey93fsn4v7l2vejcz3pf&dl=1"
  [dfd_arcface_res100]="https://www.dropbox.com/scl/fi/nn4wsf42i5kbxav93zrmh/best.pt?rlkey=lejijen0m57bbpsk9t5fnylu2&dl=1"

  # Arc2Face backend
  [arc2face_archive]="https://www.dropbox.com/scl/fi/5dcgtxoqhwy3qx1a6cfhk/scout_arc2face_assets.zip?rlkey=uf0w48wtslztbp2s7h5l6kz5y&dl=1"
)

declare -A TARGETS=(
  [ids_and_fns]="data/glint360k/ids_and_fns.npy"

  [swinface_norms]="data/glint360k/swinface_ms1mv2_swint/norms.pth"
  [swinface_templates]="data/glint360k/swinface_ms1mv2_swint/templates.pth"
  [adaface_ires50_norms]="data/glint360k/adaface_ms1mv2_ires50/norms.pth"
  [adaface_ires50_templates]="data/glint360k/adaface_ms1mv2_ires50/templates.pth"
  [arcface_res100_norms]="data/glint360k/arcface_webface42m_res100/norms.pth"
  [arcface_res100_templates]="data/glint360k/arcface_webface42m_res100/templates.pth"
  [adaface_vitb_norms]="data/glint360k/adaface_webface4m_vitb/norms.pth"
  [adaface_vitb_templates]="data/glint360k/adaface_webface4m_vitb/templates.pth"

  [clip_norms]="data/glint360k/open_clip_vitl_p14_336/norms.pth"
  [clip_templates]="data/glint360k/open_clip_vitl_p14_336/templates_f16.pth"

  [sae_swinface]="batchtopk/runs/swinface_ms1mv2_swint_k32/weights.pt"
  [sae_adaface_ires50]="batchtopk/runs/adaface_ms1mv2_ires50_k32/weights.pt"
  [sae_adaface_vitb]="batchtopk/runs/adaface_webface4m_vitb_k32/weights.pt"
  [sae_arcface_res100]="batchtopk/runs/arcface_webface42m_res100_k32/weights.pt"

  [dfd_swinface]="deep_face_decoder/runs/swinface_ms1mv2_swint_k32/checkpoints/best.pt"
  [dfd_adaface_ires50]="deep_face_decoder/runs/adaface_ms1mv2_ires50_k32/checkpoints/best.pt"
  [dfd_adaface_vitb]="deep_face_decoder/runs/adaface_webface4m_vitb_k32/checkpoints/best.pt"
  [dfd_arcface_res100]="deep_face_decoder/runs/arcface_webface42m_res100_k32/checkpoints/best.pt"

  [arc2face_archive]="downloads/scout_arc2face_assets.zip"
)

declare -A DROPBOX_PATHS=(
  [ids_and_fns]="glint360k/ids_and_fns.npy"

  [swinface_norms]="glint360k/swinface_ms1mv2_swint/norms.pth"
  [swinface_templates]="glint360k/swinface_ms1mv2_swint/templates.pth"
  [adaface_ires50_norms]="glint360k/adaface_ms1mv2_ires50/norms.pth"
  [adaface_ires50_templates]="glint360k/adaface_ms1mv2_ires50/templates.pth"
  [arcface_res100_norms]="glint360k/arcface_webface42m_res100/norms.pth"
  [arcface_res100_templates]="glint360k/arcface_webface42m_res100/templates.pth"
  [adaface_vitb_norms]="glint360k/adaface_webface4m_vitb/norms.pth"
  [adaface_vitb_templates]="glint360k/adaface_webface4m_vitb/templates.pth"

  [clip_norms]="glint360k/open_clip_vitl_p14_336/norms.pth"
  [clip_templates]="glint360k/open_clip_vitl_p14_336/templates_f16.pth"

  [sae_swinface]="checkpoints/batchtopk/swinface_ms1mv2_swint_k32/weights.pt"
  [sae_adaface_ires50]="checkpoints/batchtopk/adaface_ms1mv2_ires50/weights.pt"
  [sae_adaface_vitb]="checkpoints/batchtopk/adaface_webface4m_vitb_k32/weights.pt"
  [sae_arcface_res100]="checkpoints/batchtopk/arcface_ms1mv3_r50_k32/weights.pt"

  [dfd_swinface]="checkpoints/deep_face_decoder/swinface_ms1mv2_swint_k32/best.pt"
  [dfd_adaface_ires50]="checkpoints/deep_face_decoder/adaface_ms1mv2_ir50_k32/best.pt"
  [dfd_adaface_vitb]="checkpoints/deep_face_decoder/adaface_webface4m_vitb_k32/best.pt"
  [dfd_arcface_res100]="checkpoints/deep_face_decoder/arcface_ms1mv3_r50_k32/best.pt"

  [arc2face_archive]="checkpoints/scout_arc2face_assets.zip"
)

ids_keys() {
  printf '%s\n' ids_and_fns
}

fr_template_keys() {
  printf '%s\n' \
    swinface_norms \
    swinface_templates \
    adaface_ires50_norms \
    adaface_ires50_templates \
    arcface_res100_norms \
    arcface_res100_templates \
    adaface_vitb_norms \
    adaface_vitb_templates
}

clip_keys() {
  printf '%s\n' clip_norms clip_templates
}

sae_keys() {
  printf '%s\n' \
    sae_swinface \
    sae_adaface_ires50 \
    sae_adaface_vitb \
    sae_arcface_res100
}

dfd_keys() {
  printf '%s\n' \
    dfd_swinface \
    dfd_adaface_ires50 \
    dfd_adaface_vitb \
    dfd_arcface_res100
}

arc2face_keys() {
  printf '%s\n' arc2face_archive
}

keys_for_group() {
  case "$1" in
    ids) ids_keys ;;
    fr-templates|fr_templates) fr_template_keys ;;
    clip|clip-templates|clip_templates) clip_keys ;;
    templates) ids_keys; fr_template_keys; clip_keys ;;
    sae|sae-weights|sae_weights) sae_keys ;;
    dfd|dfd-weights|dfd_weights) dfd_keys ;;
    arc2face|arc2face-weights|arc2face_weights) arc2face_keys ;;
    all) ids_keys; fr_template_keys; clip_keys; sae_keys; dfd_keys; arc2face_keys ;;
    *) die "unknown asset group '$1'. Run: bash download_assets.sh list" ;;
  esac
}

valid_group() {
  case "$1" in
    ids|fr-templates|fr_templates|clip|clip-templates|clip_templates|templates|sae|sae-weights|sae_weights|dfd|dfd-weights|dfd_weights|arc2face|arc2face-weights|arc2face_weights|all)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

list_files() {
  for group in ids fr-templates clip sae dfd arc2face; do
    echo "[$group]"
    while IFS= read -r key; do
      printf '  %-24s %s -> %s\n' "$key" "${DROPBOX_PATHS[$key]}" "${TARGETS[$key]}"
    done < <(keys_for_group "$group")
  done
}

download_one() {
  local key="$1"
  local url="${URLS[$key]}"
  local target="$REPO_ROOT/${TARGETS[$key]}"

  [[ -n "$url" ]] || die "missing Dropbox URL for '$key' -> $target"
  mkdir -p "$(dirname "$target")"

  if [[ -s "$target" ]]; then
    echo "exists: $target"
    if [[ "$key" == "arc2face_archive" ]]; then
      extract_archive "$target"
    fi
    return
  fi

  echo "download: $target"
  local tmp="${target}.part"
  wget -nv -c -O "$tmp" "$url"
  mv "$tmp" "$target"

  if [[ "$key" == "arc2face_archive" ]]; then
    extract_archive "$target"
  fi
}

extract_archive() {
  local archive="$1"
  command -v unzip >/dev/null 2>&1 || die "unzip is required to extract $archive"
  echo "extract: $archive -> $REPO_ROOT"
  unzip -q -o "$archive" -d "$REPO_ROOT"
}

main() {
  if [[ $# -eq 0 ]]; then
    usage
    exit 1
  fi

  if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ "$1" == "list" ]]; then
    list_files
    exit 0
  fi

  local requested=()
  local seen=" "
  local key
  local group

  for group in "$@"; do
    valid_group "$group" || die "unknown asset group '$group'. Run: bash download_assets.sh list"
    while IFS= read -r key; do
      if [[ "$seen" != *" $key "* ]]; then
        requested+=("$key")
        seen+="$key "
      fi
    done < <(keys_for_group "$group")
  done

  for key in "${requested[@]}"; do
    download_one "$key"
  done
}

main "$@"
