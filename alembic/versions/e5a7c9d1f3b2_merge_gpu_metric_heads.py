"""Merge the GPU-provider and evaluation-contract migration branches.

Revision ID: e5a7c9d1f3b2
Revises: 2ab3c4d5e6f7, d4a2b1c50e67
Create Date: 2026-08-30
"""

from __future__ import annotations


revision = "e5a7c9d1f3b2"
down_revision = ("2ab3c4d5e6f7", "d4a2b1c50e67")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Both parent branches already contain their complete schema changes."""


def downgrade() -> None:
    """Split the history back to the two parent heads without changing data."""
