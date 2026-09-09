REPO: $REPO
PR NUMBER: $PR
BRANCH: $BRANCH

A reviewer left comments on this PR. The branch is checked out.

List every review comment:
`gh api repos/$REPO/pulls/$PR/comments --paginate` and `gh pr view $PR --comments`

Every reply you post starts with the line `🔨 **Implementer**` followed by a
blank line. You are replying inside the reviewer's own threads under the same
`github-actions[bot]` account, so that header is the only thing separating your
reply from the comment it answers.

Do one of two things with each comment.

- Fix it, then reply to that exact thread saying what you changed:
  `gh api repos/$REPO/pulls/$PR/comments/<comment_id>/replies -f body='...'`
- Push back. If the comment is wrong, or does not apply to this code, reply
  explaining concretely why and change nothing. Disagreeing is a valid outcome.
  Do not edit code to satisfy a comment you believe is mistaken, and do not
  widen the change beyond what the comment asks for.

Every comment gets a reply. Then commit and push to the same branch. Do not open
a new PR, and do not force-push.

Re-run whatever tests cover the code you touched.
