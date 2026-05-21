Local sim_engine customizations (qwen classifier streaming fix, object-only
lookup, per-object debug colour) are kept here as a patch instead of being
committed to the sim_engine submodule.

Auto-reapply: git hooks (post-merge / post-checkout / post-rewrite) in the
submodule reapply this patch after every pull/checkout, so pulls don't break
your setup.

Manual reapply (if ever needed), from the project root:
  git -C realworld/src/sim_engine apply patches/sim_engine_qwen.patch

Refresh the patch after making NEW local edits to the submodule:
  ( cd realworld/src/sim_engine && git diff ) > patches/sim_engine_qwen.patch

Model choice is separate and already pull-proof: set the OLLAMA_MODEL env var.
