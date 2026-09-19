"""
CheckpointManager - Save and resume trial results for crash recovery.

Stores completed trial results to disk so that if the process crashes,
it can resume from the last completed trial without re-running previous trials.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
from datetime import datetime

from Muffakir.exceptions import ConfigurationError, CorruptCheckpointError

from .results.trial import TrialResult

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint storage for architecture search trials.
    
    Saves completed trial results to a JSON file so that the search
    can be resumed after a crash or interruption.
    
    Writes are atomic (temp file + os.replace) so a crash mid-write never
    corrupts the existing checkpoint — exactly the scenario this feature
    exists to survive.
    
    Attributes:
        checkpoint_dir: Directory to store checkpoint files
        checkpoint_file: Path to the main checkpoint JSON file
    """
    
    def __init__(self, checkpoint_dir: str = "./muffakir_checkpoints/"):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory path for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / "composer_checkpoint.json"
        self._ensure_directory()

        # In-memory mirror of the checkpoint file, populated lazily on first
        # access. Without this, save_trial()/save_trials_batch() each did a
        # full disk-read-and-reparse of every existing trial before rewriting
        # the file — a redundant O(n) pass on top of the O(n) write, every
        # single trial completion across a run (O(n^2) total for n trials).
        # None means "not yet loaded from disk this instance's lifetime".
        self._trials_cache: Optional[Dict[int, TrialResult]] = None
        self._search_space_cache: Optional[Dict[str, Any]] = None
        self._pricing_snapshot_cache: Optional[Dict[str, Any]] = None
        self._config_hash_cache: Optional[str] = None

    def _ensure_directory(self):
        """Create checkpoint directory if it doesn't exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        """Populate the in-memory trial/search-space cache from disk, once."""
        if self._trials_cache is not None:
            return

        if not self.checkpoint_file.exists():
            self._trials_cache = {}
            self._search_space_cache = None
            self._pricing_snapshot_cache = None
            self._config_hash_cache = None
            return

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            trials = [TrialResult.from_dict(t) for t in data.get("trials", [])]
            self._trials_cache = {t.trial_id: t for t in trials}
            self._search_space_cache = data.get("search_space")
            self._pricing_snapshot_cache = data.get("pricing_snapshot")
            self._config_hash_cache = data.get("config_hash")
            logger.debug(f"Loaded {len(trials)} trials from checkpoint")

        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
            raise CorruptCheckpointError(
                f"Checkpoint file is corrupt: {self.checkpoint_file} ({e})",
                context={"checkpoint_file": str(self.checkpoint_file)},
            ) from e

    def _write_current_state(self) -> None:
        """Serialize the in-memory cache to disk (atomic write)."""
        checkpoint_data: Dict[str, Any] = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "trials": [t.to_dict() for t in self._trials_cache.values()],
        }
        if self._search_space_cache is not None:
            checkpoint_data["search_space"] = self._search_space_cache
        if self._pricing_snapshot_cache is not None:
            checkpoint_data["pricing_snapshot"] = self._pricing_snapshot_cache
        if self._config_hash_cache is not None:
            checkpoint_data["config_hash"] = self._config_hash_cache
        self._write_atomic(checkpoint_data)
    
    def _write_atomic(self, checkpoint_data: dict) -> None:
        """Atomically write checkpoint data (temp file + os.replace)."""
        # Ensure directory exists (path may have been removed between calls)
        self._ensure_directory()
        fd, tmp_path = tempfile.mkstemp(
            prefix=".composer_checkpoint_",
            suffix=".tmp",
            dir=str(self.checkpoint_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(tmp_path, self.checkpoint_file)
        except Exception as e:
            logger.error(f"Checkpoint write failed: {e}", exc_info=True)
            # Clean up the temp file if the write/replace failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    
    def save_trial(self, trial: TrialResult, search_space: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a completed trial to the checkpoint file.

        Replaces an existing entry with the same trial_id (idempotent) and writes
        atomically so a crash never corrupts the checkpoint.

        Args:
            trial: The completed TrialResult to save
            search_space: The search space this trial's ID was generated from.
                Stored once and carried forward on every subsequent save (pass
                it every call — cheap, small, constant per run) so a resumed
                run can detect a search space change via ``validate_search_space``.
        """
        self._ensure_loaded()
        self._trials_cache[trial.trial_id] = trial
        if search_space is not None:
            self._search_space_cache = search_space

        self._write_current_state()
        logger.debug(f"Checkpoint saved: Trial {trial.trial_id}")

    def save_trials_batch(
        self, trials: List[TrialResult], search_space: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save multiple trials to checkpoint at once.

        More efficient than calling save_trial() multiple times. De-duplicates by
        trial_id and writes atomically.

        Args:
            trials: List of TrialResults to save
            search_space: See ``save_trial``.
        """
        if not trials:
            return

        self._ensure_loaded()
        for trial in trials:
            self._trials_cache[trial.trial_id] = trial  # add/replace
        if search_space is not None:
            self._search_space_cache = search_space

        self._write_current_state()
        logger.info(f"Checkpoint saved: {len(trials)} trials batch")

    def load_completed_trials(self) -> List[TrialResult]:
        """
        Load all completed trials from checkpoint.

        Returns:
            List of TrialResult objects from checkpoint, empty list if no checkpoint
            file exists yet.

        Raises:
            CorruptCheckpointError: If the checkpoint file exists but is malformed.
                A missing file and a corrupt file are different failure modes -
                the former is normal (fresh run), the latter risks silently
                discarding prior trial results if treated as "no checkpoint".
                Recover via ``clear()``.
        """
        self._ensure_loaded()
        return list(self._trials_cache.values())

    def get_stored_search_space(self) -> Optional[Dict[str, Any]]:
        """Return the search space stored in the checkpoint, or None if none was ever saved."""
        self._ensure_loaded()
        return self._search_space_cache

    def set_search_space(self, search_space: Dict[str, Any]) -> None:
        """
        Persist *search_space* immediately, even if no trial has completed yet.

        Call once at the start of a run (after ``validate_search_space``) so the
        search space this run is using is recorded even if the process crashes
        before the first trial finishes — a later resume can then still detect
        a search-space change.
        """
        self._ensure_loaded()
        self._search_space_cache = search_space
        self._write_current_state()

    def validate_config_hash(self, current: str) -> None:
        """Reject resume when any run-wide configuration has changed.

        Historical checkpoints without this field remain readable and rely on
        the older search-space-only validation.
        """
        self._ensure_loaded()
        if self._config_hash_cache is not None and self._config_hash_cache != current:
            raise ConfigurationError(
                "Run configuration changed since the last checkpoint at "
                f"'{self.checkpoint_file}'. This includes per-run prompt changes. "
                "Pass resume=False, or use a different checkpoint_dir."
            )

    def set_config_hash(self, config_hash: str) -> None:
        self._ensure_loaded()
        self._config_hash_cache = config_hash
        self._write_current_state()

    def get_stored_pricing_snapshot(self) -> Optional[Dict[str, Any]]:
        """Return the pricing snapshot stored in the checkpoint, or None if none was ever saved."""
        self._ensure_loaded()
        return self._pricing_snapshot_cache

    def set_pricing_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Persist a PriceMap snapshot (see Pricing.price_map.PriceMap.to_dict())
        immediately, even if no trial has completed yet.

        Call once per fresh fit() run, right after fetching prices, so a
        crash before the first trial finishes still leaves the exact fetched
        snapshot on disk -- a later resume reuses it instead of re-fetching.
        """
        self._ensure_loaded()
        self._pricing_snapshot_cache = snapshot
        self._write_current_state()

    def validate_search_space(self, current: Dict[str, Any]) -> None:
        """
        Raise if *current* differs from the search space stored in the checkpoint.

        Trial IDs are positional indices over the search space
        (``ConfigSpace.generate_combinations_with_ids``): if the search space
        changes between two ``fit()`` runs resuming from the same
        ``checkpoint_dir``, a recycled trial_id can silently be treated as
        "already completed" for a completely different configuration. Call
        this before resuming to fail fast instead.

        Args:
            current: The search space about to be used for this run.

        Raises:
            ConfigurationError: If a search space was previously stored and
                differs from *current*.
        """
        stored = self.get_stored_search_space()
        if stored is not None and stored != current:
            raise ConfigurationError(
                "Search space changed since the last checkpoint at "
                f"'{self.checkpoint_file}'. Resuming would silently skip newly "
                "added trials whose IDs collide with previously completed ones. "
                "Pass resume=False, or use a different checkpoint_dir."
            )
    
    def get_completed_ids(self) -> Set[int]:
        """
        Get set of trial IDs that have been completed.
        
        Returns:
            Set of trial IDs from checkpoint
        """
        trials = self.load_completed_trials()
        return {t.trial_id for t in trials}
    
    def get_trial_count(self) -> int:
        """
        Get number of completed trials in checkpoint.
        
        Returns:
            Number of trials saved in checkpoint
        """
        return len(self.load_completed_trials())
    
    def has_checkpoint(self) -> bool:
        """
        Check if a checkpoint file exists.
        
        Returns:
            True if checkpoint file exists
        """
        return self.checkpoint_file.exists()
    
    def clear(self) -> None:
        """
        Clear the checkpoint file for a fresh run.

        Deletes the checkpoint file if it exists.
        """
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")
        self._trials_cache = None
        self._search_space_cache = None
        self._pricing_snapshot_cache = None
        self._config_hash_cache = None

    def get_last_updated(self) -> Optional[datetime]:
        """
        Get timestamp of last checkpoint update.

        Returns:
            datetime of last update, or None if no checkpoint file exists yet.

        Raises:
            CorruptCheckpointError: If the checkpoint file exists but is malformed
                or contains an unparseable timestamp. See load_completed_trials().
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            last_updated = data.get("last_updated")
            if last_updated:
                return datetime.fromisoformat(last_updated)
            return None

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise CorruptCheckpointError(
                f"Checkpoint file is corrupt: {self.checkpoint_file} ({e})",
                context={"checkpoint_file": str(self.checkpoint_file)},
            ) from e

    def __repr__(self) -> str:
        count = self.get_trial_count()
        return f"CheckpointManager(trials={count}, path={self.checkpoint_file})"
