# How the raw reply breaks the enforcement policy

Reply checked: [ana_protest_raw_reply.txt](ana_protest_raw_reply.txt)
Policy: [Violent Content Enforcement Actions and Policy Reasons.md](../Violent%20Content%20Enforcement%20Actions%20and%20Policy%20Reasons.md)

| # | What the reply does | Policy rule it breaks | Result |
|---|---|---|---|
| 1 | States no policy reason at all. | "Every decision must state its reason using the approved wording." A decision that "states no reason" is returned unenforced. | Unenforceable |
| 2 | Each evidence moment is a single point (`[00:30]`, `[01:50]`), with no end. | Each moment needs a **Start** and an **End**, and "the start must come before the end". A point has start equal to end, which the policy refuses explicitly. | Unenforceable |
| 3 | Times are written as `mm:ss` inside brackets. | Start and End are "seconds from the start of the clip". | Wrong format (must be converted) |
| 4 | The action is buried in free text (`Label: age-restrict`) rather than a dedicated field. | Every decision "carries exactly one action". The value itself is valid (lowercase, hyphenated), but nothing guarantees the model won't write `restricted`, `flag` or two actions next time. | Not reliably enforceable |
| 5 | The reply is unstructured prose. | Reviewers and the queue need action, reason and moments as separate parts. | Not machine-checkable |

What the reply gets right:

- `age-restrict` is a permitted action, written exactly as the queue shows it.
- It cites at least one moment, as `age-restrict` requires.
- Both descriptions are one plain sentence saying what is shown.
- Both moments (30s and 110s) lie within the clip's 198-second runtime from the upload record.

The correct reason for this decision would be **VC-03, "Graphic violence shown in a news or documentary context"**, the only approved reason for `age-restrict` that fits news footage.
