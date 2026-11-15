# Violent Content Enforcement Actions and Policy Reasons

This is a one-page excerpt of YouTube's Violent and Graphic Content enforcement policy. It is kept by Lena Hofmann, Violent Content Policy Lead, Content Policy. It has been in effect since 2024-11-04 and is current as of 2025-03-10.

## Enforcement actions

Every decision carries exactly one action. Actions are written exactly as the reviewer queue shows them: all lowercase, with a hyphen in `age-restrict`.

| Action | When it applies |
|---|---|
| `keep` | The video stays up with no restriction. Violence is absent, incidental or non-graphic. |
| `age-restrict` | The video stays up for signed-in viewers aged 18 and over only. It is used for graphic violence that has news, documentary, educational or artistic context. |
| `remove` | The video is taken down. It is used for violence posted to glorify, incite or threaten, and for graphic violence with no context. |

## Approved policy reasons

Every decision must state its reason using the approved wording below, exactly as written. Reviewers read the reason text, not just the code. A code given without its wording does not count as a stated reason. Free-text or paraphrased reasons are not accepted.

| Reason Code | Reason Wording | Permitted Actions |
|---|---|---|
| VC-00 | No violent content requiring enforcement | keep |
| VC-01 | Non-graphic or incidental violence | keep |
| VC-02 | Graphic violence with educational or artistic context | age-restrict |
| VC-03 | Graphic violence shown in a news or documentary context | age-restrict |
| VC-04 | Graphic violence without context | remove |
| VC-05 | Violence posted to glorify or celebrate harm | remove |
| VC-06 | Incitement or credible threat of violence | remove |

## Evidence a decision must cite

Evidence is cited as moments from the clip.

| Action | Moments required |
|---|---|
| `age-restrict` | At least one |
| `remove` | At least one |
| `keep` | None required. A keep decision may cite moments that support keeping, such as non-graphic footage. |

Each moment has three parts:

- **Start**: seconds from the start of the clip.
- **End**: seconds from the start of the clip. The start must come before the end, so the span has positive length.
- **Description**: one plain sentence saying what is shown or said, enough for a reviewer to know what to look for when they jump to it.

A moment must lie within the clip. A reviewer cannot check a span that ends after the clip ends, so that span is invalid.

## Decisions that cannot be enforced

A decision in any of the cases below is returned to the queue unenforced:

- It states no reason, or it states a reason that is not one of the seven approved wordings.
- Its action is anything other than `keep`, `age-restrict` or `remove`. Examples that are refused: `flag`, `restricted`, `review`.
- It is an `age-restrict` or `remove` decision that cites no moments.
- A cited moment does not start before it ends. This includes a start equal to its end.
- A cited moment falls outside the clip's runtime as recorded on the upload. The runtime always comes from the upload record and never from the decision itself.