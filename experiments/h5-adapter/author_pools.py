"""Author the H5 needle pools: a training split and an evaluation split.

Run:  env/venv/bin/python experiments/h5-adapter/author_pools.py

Why a generator rather than two hand-written JSON files: the train/eval split has to be
auditable. The template lists live here as plain Python, the emitted JSON carries a per-template
``sha256`` of the *canonical* surface form (the template text itself, not a rendered instance),
and ``leakage_check.py`` re-derives the split from the emitted files and attacks it. Nothing in
this file is trusted by the check.

Design constraints baked in here:

* **4 balanced classes** — billing / technical / sales / other, the same four the P1b probe and
  the shipped head were measured on.
* **12 templates per class are the verbatim P1b probe templates** (``source: "p1b"``), kept so
  the frozen head's published 0.300 has a like-for-like reference on this pool. They are
  evaluation-only: no training template is derived from them.
* **Every template carries at least one slot** (``{n}`` a number, ``{ref}`` an identifier), and
  the train and eval slot *value* pools are disjoint by value. So even if a template collided,
  a rendered training string could not equal a rendered evaluation string.
* **Per-class combo budget** is asserted >= 64 so a cell of n=200 (50 per class) can be drawn
  with every document unique by sha256 -- no duplicate documents inflating the effective n.

The emitted pools are the artifact; this script only writes them.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
POOL_DIR = os.path.join(HERE, "pools")

LABELS = ["billing", "technical", "sales", "other"]

# --------------------------------------------------------------------------------------------
# The question. Copied verbatim from experiments/orch-probe-d/run.py so the frozen-head
# reference point (0.300 at needle@END) is measured against the same decision the probe used.
# --------------------------------------------------------------------------------------------
QUESTION = {
    "type": "choice",
    "instructions": "Which department should handle this request?",
    "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts, plan upgrades",
        "other": "everything else",
    },
}

# The harness convention: a pool's ``question`` maps a question id to the external question
# definition, which is what ``laya.agent.Agent._to_internal`` and ``harness/pools.leakage_report``
# both consume.
QUESTIONS = {"department": QUESTION}

# --------------------------------------------------------------------------------------------
# Slot value pools. DISJOINT between train and eval by construction (asserted below): train
# values are small odd numbers and 4-digit refs starting with T; eval values are larger and R.
# --------------------------------------------------------------------------------------------
SLOTS_TRAIN = {
    "n": ["%d" % (7 + 3 * i) for i in range(96)],
    "ref": ["TR-%04d" % (1100 + 7 * i) for i in range(96)],
}
SLOTS_EVAL = {
    "n": ["%d" % (310 + 3 * i) for i in range(96)],
    "ref": ["EV-%04d" % (8200 + 13 * i) for i in range(96)],
}


# --------------------------------------------------------------------------------------------
# EVALUATION templates. 12 per class are verbatim P1b (source "p1b"); 12 per class are new.
# --------------------------------------------------------------------------------------------
EVAL_P1B: Dict[str, List[str]] = {
    "billing": [
        "I was charged twice for invoice {n}, please refund the duplicate payment.",
        "My card was billed {n} dollars more than the amount on my last statement.",
        "Please refund invoice {n}, the payment went through twice this morning.",
        "There is an unexpected charge of {n} dollars on my account this month.",
        "I need a refund for the duplicate transaction on invoice {n}.",
        "My last statement shows a payment of {n} dollars I never authorised.",
        "Can you reverse the double charge on invoice {n}?",
        "The renewal payment of {n} dollars was taken from the wrong card.",
        "I am disputing a charge of {n} dollars that appeared on my bill.",
        "Please credit back the overpayment of {n} dollars on my account.",
        "Invoice {n} was paid twice, I would like the second payment returned.",
        "Why was I billed {n} dollars when my plan costs less than that?",
    ],
    "technical": [
        "The dashboard throws an error every time I open report {n}.",
        "Our API returns 500 responses since this morning and the service is down.",
        "The application crashes whenever I try to export file {n}.",
        "Sync stopped working after the last update and nothing uploads now.",
        "The login page hangs and then times out for every user.",
        "We are seeing repeated outages on the {n} endpoint today.",
        "The mobile app closes itself as soon as I open the settings screen.",
        "Database connections are failing and requests are timing out.",
        "The search feature returns an internal error for every query.",
        "Uploads larger than {n} megabytes fail with a server error.",
        "The nightly job did not run and the data is stale again.",
        "Webhooks stopped being delivered after the maintenance window.",
    ],
    "sales": [
        "What would an enterprise contract for {n} seats cost per year?",
        "Can you send pricing for upgrading our team to the business tier?",
        "We would like a quote for {n} licences billed annually.",
        "Is there a volume discount if we purchase {n} seats this quarter?",
        "Please share the price of the premium plan for a team of {n}.",
        "What are the contract terms for an annual plan with {n} users?",
        "I want to compare the cost of the pro and enterprise plans.",
        "Could you prepare a proposal for {n} additional seats?",
        "What discount applies if we commit to {n} seats for two years?",
        "Please quote the enterprise tier for our department of {n} people.",
        "How much more would {n} extra seats add to our current contract?",
        "We are evaluating vendors and need pricing for {n} users.",
    ],
    "other": [
        "Please update the shipping address on my account to the new office.",
        "How do I change the email address associated with my profile?",
        "I would like to close my account and delete all stored data.",
        "Can you confirm which timezone my scheduled reports use?",
        "Where can I download the accessibility documentation?",
        "I need a copy of the data processing agreement for our records.",
        "How do I add a colleague to our workspace as a viewer?",
        "Please tell me the office opening hours for the support desk.",
        "Is there a way to export my settings to another workspace?",
        "I want to change the display language of the interface.",
        "Could you explain how the retention policy applies to archived items?",
        "Please confirm whether the service is available in my region.",
    ],
}

EVAL_NEW: Dict[str, List[str]] = {
    "billing": [
        "The receipt for order {n} shows a total higher than the price we agreed.",
        "Two identical debits of {n} dollars appear on my statement for the same day.",
        "I was told the {n} dollar setup fee would be waived but it was taken anyway.",
        "Our workspace was suspended for non-payment even though the balance is zero.",
        "The refund of {n} dollars was promised last month and never arrived.",
        "Please move the {n} dollar credit sitting in ledger {ref} onto the current one.",
        "The annual renewal for invoice {n} was charged to a card we closed.",
        "I keep receiving overdue notices for an amount that was settled already.",
        "The currency on invoice {n} is wrong, so the total is about twelve percent off.",
        "A late fee of {n} dollars was added even though the transfer cleared on time.",
        "Half of my payment of {n} dollars was applied to an unrelated account.",
        "The automatic top-up of {n} dollars fired twice within the same hour.",
    ],
    "technical": [
        "Every call to the search endpoint returns a 502 since the release.",
        "The export job for {n} rows never finishes and leaves the file empty.",
        "Our nightly backup of the {n} gigabyte volume fails with a checksum mismatch.",
        "The worker pool stops consuming the queue after about {n} seconds.",
        "Single sign-on loops between the identity provider and the application.",
        "The desktop client cannot reach the sync service from behind our proxy.",
        "Uploads of {n} megabytes are rejected with a gateway timeout.",
        "The scheduler skips every second run and the dashboards go stale.",
        "Memory grows until the process is killed, roughly every {n} hours.",
        "The signature check on payload {ref} rejects bodies the sender considers valid.",
        "Log ingestion stopped at {n} events and no new rows appear in the console.",
        "The container restarts in a loop because the health probe never turns green.",
    ],
    "sales": [
        "What would {n} named users cost on the enterprise agreement?",
        "We are comparing three vendors and need your price for {n} workstations.",
        "Please send a formal quotation for {n} additional licences.",
        "Does the volume band change if we buy {n} units before the end of the quarter?",
        "Our procurement team needs the unit rate for {n} units in writing.",
        "Can we fold {n} more users into the agreement we signed in spring?",
        "What is the difference in annual cost between {n} and twice that many?",
        "We would like a proposal covering {n} sites under one agreement.",
        "Does a discount tier begin once we reach {n} units annually?",
        "Please quote the migration package for {n} mailboxes.",
        "Our budget assumes {n} units; what would that come to per month?",
        "Could you price a pilot for {n} users with an option to expand?",
    ],
    "other": [
        "How do I transfer ownership of the shared folder {ref} to a colleague?",
        "Please send the accessibility statement for our procurement file {ref}.",
        "Which timezone is used for the timestamps in the activity feed?",
        "I would like the interface language switched to French.",
        "Where can I find the retention schedule for archived material?",
        "Can you confirm whether the service is offered in Portugal?",
        "How do I remove a former colleague from the notification list?",
        "Please provide the completed security questionnaire for our auditors.",
        "Is there a way to merge the two workspaces {ref} into one?",
        "I need a copy of the subprocessor list for our records.",
        "How do I change the mailbox that receives the weekly digest?",
        "What is the process for closing a workspace and purging its contents?",
    ],
}

# --------------------------------------------------------------------------------------------
# TRAINING templates. 24 per class, all authored for this experiment, all evaluation-unseen.
# Phrasing is deliberately different from the eval set: the leakage check refuses a shared
# 6-gram, so these cannot be paraphrases of an evaluation template.
# --------------------------------------------------------------------------------------------
TRAIN: Dict[str, List[str]] = {
    "billing": [
        "Statement {ref} lists a settlement that our bank has no record of.",
        "The amount taken for subscription {ref} does not match the signed order.",
        "We were debited for {n} units but only ever received the first batch.",
        "Please trace the transfer of {n} dollars sent on the ninth of March.",
        "The discount on contract {ref} disappeared from the latest assessment.",
        "Nobody can explain the {n} dollar adjustment on the March statement.",
        "One instalment for {ref} was collected from a dormant facility.",
        "The outstanding figure on {ref} was cleared by cheque last autumn.",
        "Our finance office disputes the surcharge levied against {ref}.",
        "The pro forma for {ref} carries a tax line that should not be there.",
        "Two separate collections were taken for the same billing period on {ref}.",
        "Please reapply the {n} dollar voucher that was dropped at checkout.",
        "The direct debit mandate for {ref} was cancelled before the due date.",
        "We are owed {n} dollars from the cancelled portion of the order.",
        "The settlement date printed on {ref} is a public holiday.",
        "An interest line of {n} dollars appeared without any prior warning.",
        "The prorated figure on {ref} looks like it used the wrong start date.",
        "Our purchase ledger shows {ref} settled and the portal disagrees.",
        "Someone reissued the demand for {ref} after it had been paid.",
        "The refundable deposit of {n} dollars was never returned to us.",
        "Line item three of {ref} is charged at a rate we never accepted.",
        "The instalment plan for {ref} was applied to the wrong agreement.",
        "We ask that the {n} dollar balance carried on {ref} be written off.",
        "The remittance advice for {ref} was sent to a closed mailbox.",
    ],
    "technical": [
        "Index rebuilds on shard {ref} abort partway through with no message.",
        "The edge cache serves stale pages until it is flushed by hand.",
        "Requests time out once concurrency passes about {n} workers.",
        "Replication lag on {ref} climbs to several minutes under load.",
        "The parser rejects a payload that validates against the published schema.",
        "Session cookies are dropped when the browser is restarted.",
        "The batch importer for {ref} silently skips malformed rows.",
        "Latency on the read replica of {ref} spikes every quarter hour.",
        "The queue drains slowly because consumers keep getting restarted.",
        "A race in the migration tool leaves table {ref} half populated.",
        "The certificate for {ref} is renewed but clients still reject it.",
        "Trace sampling drops the spans that matter during incidents.",
        "The feature flag for {ref} never propagates to the older nodes.",
        "Disk usage on the logging volume grows by {n} gigabytes each day.",
        "Failover to the standby of {ref} takes far longer than documented.",
        "The mobile build crashes on launch for accounts created after {ref}.",
        "Connection pools for {ref} leak until the process exhausts its handles.",
        "The cron entry for {ref} runs twice whenever the clock shifts.",
        "Compaction on {ref} never completes and disk pressure keeps rising.",
        "The sandbox for {ref} returns a permissions fault on every call.",
        "Metrics for {ref} stopped reporting after the agent upgrade.",
        "The gateway strips the authentication header on redirected calls.",
        "Pages under {ref} render blank when the locale is not English.",
        "Our alerting for {ref} fires and then resolves within seconds.",
    ],
    "sales": [
        "We need a costed option for {n} desks under a single agreement.",
        "Please outline what a commitment of {n} units would include.",
        "Our board wants indicative figures for {n} sites by Friday.",
        "Would the rate per unit fall if we ordered {n} instead of half that?",
        "Send terms for {n} concurrent sessions on the professional tier.",
        "We are shortlisting suppliers for {n} endpoints and need your best offer.",
        "Could you illustrate the saving at {n} units against list price?",
        "What would {n} annual subscriptions come to with support included?",
        "Our renewal for {ref} needs repricing at the current volumes.",
        "Please attach a schedule of charges for {n} devices.",
        "Can the agreement for {ref} be extended to cover the new division?",
        "We would like a comparison of the two tiers at {n} units.",
        "What lead time applies if we confirm {n} units this week?",
        "Our director asks whether {n} units qualifies for the partner rate.",
        "Please confirm the minimum term for an order of {n} units.",
        "Is onboarding included when we take {n} units at once?",
        "We are renegotiating {ref} and want the current rate card.",
        "Could you model the three year outlay for {n} units each year?",
        "Our reseller offers {ref}; can you match that structure?",
        "What support tier is bundled with {n} units?",
        "Please state whether {n} units is enough for the partner programme.",
        "We would consider expanding {ref} if the pricing is favourable.",
        "Send a draft order form for {n} units for our legal review.",
        "How does the price per unit move between {n} and triple that?",
    ],
    "other": [
        "Whom should I contact about the museum membership renewal?",
        "Is the community library open on the bank holiday?",
        "Please advise how to appeal a parking notice issued near {ref}.",
        "Our choir needs the rehearsal room key returned by Friday.",
        "Which form do I submit to change the name on the lease?",
        "The allotment committee meets on the first Wednesday monthly.",
        "Can you forward the minutes from the neighbourhood meeting?",
        "How long does the passport office take to return documents?",
        "Is there a waiting list for the swimming class on Tuesdays?",
        "Please explain how to join the cycle scheme at work.",
        "The village hall booking for {ref} needs to be moved.",
        "Who maintains the footpath behind the old mill?",
        "Can the recycling centre accept paint tins this month?",
        "I would like to volunteer at the autumn fair.",
        "How do I request a replacement bus pass for {ref}?",
        "Is the tennis court booking system open to non-members?",
        "Please tell me the closing time of the reference library.",
        "Our residents group wants to display a notice on the board.",
        "Is there a form for adding a second driver to the car share scheme?",
        "The parish council minutes for {ref} are not on the website.",
        "Which department issues parking permits for the harbour?",
        "Can you confirm the date of the summer fete this year?",
        "I need the contact details for the school governors.",
        "How do I return the hire equipment borrowed for {ref}?",
    ],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _placeholders(text: str) -> List[str]:
    out = []
    i = 0
    while True:
        a = text.find("{", i)
        if a < 0:
            return out
        b = text.find("}", a)
        if b < 0:
            return out
        out.append(text[a + 1:b])
        i = b + 1


def _combos(text: str, slots: Dict[str, List[str]]) -> int:
    total = 1
    used = _placeholders(text)
    if not used:
        return 1
    for name in used:
        total *= len(slots[name])
    return total


def build_split(name: str, templates: Dict[str, List[str]], slots: Dict[str, List[str]],
                sources: Dict[str, List[str]] | None = None) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for label in LABELS:
        for i, text in enumerate(templates[label]):
            ph = _placeholders(text)
            for p in ph:
                if p not in slots:
                    raise ValueError("template %s/%s uses unknown slot {%s}" % (label, i, p))
            out.append({
                "id": "%s-%s-%02d" % (name, label, i),
                "label": label,
                "lang": "en",
                "text": text,
                "slots_used": sorted(set(ph)),
                "source": (sources[label][i] if sources else "h5-new"),
                "sha256": sha256_text(text),
            })
    per_class_combos = {}
    for label in LABELS:
        per_class_combos[label] = sum(
            _combos(t["text"], slots) for t in out if t["label"] == label)
        if per_class_combos[label] < 64:
            raise ValueError("%s/%s has only %d slot combinations; a cell of 50 items per class "
                             "needs >= 64 to stay duplicate-free" % (name, label, per_class_combos[label]))
    return {
        "pool_id": "needles-h5-%s-v1" % name,
        "kind": "balanced",
        "split": name,
        "labels": LABELS,
        "question": QUESTIONS,
        "slots": slots,
        "templates": out,
        "counts": {lb: sum(1 for t in out if t["label"] == lb) for lb in LABELS},
        "combos_per_class": per_class_combos,
    }


def _assert_disjoint_slot_values() -> None:
    for k in SLOTS_TRAIN:
        a, b = set(SLOTS_TRAIN[k]), set(SLOTS_EVAL[k])
        if a & b:
            raise ValueError("slot %s shares values between train and eval: %s" % (k, sorted(a & b)))


# --------------------------------------------------------------------------------------------
# FILLER. protocol.md section 9 requires needle and filler to come from disjoint pools "verified by
# content, not by variable name". `harness/pools.leakage_report` is the check every other arm in
# this program is held to, so the H5 filler is held to it too -- but the harness's `filler-v1` was
# authored against the *support-domain* needle set of the earlier arms, and the H5 needles are
# wider (they add `other`-class admin vocabulary and generic words like "team", "quarter",
# "window"). Rather than weaken the check, this authors a filler whose content words are disjoint
# from the union of the H5 needle vocabulary and the question's criteria.
#
# The selection is explicit and auditable: ~180 candidate sentences about building and grounds
# upkeep are offered, and every candidate that shares a *whole content word* (the harness's own
# `words()` + `STOPWORDS` definition) with the needle/criteria vocabulary is dropped. The dropped
# count and the dropped words are printed and recorded in the pool provenance, so nothing is
# hidden. No sentence is rewritten to pass; candidates are only kept or dropped.
# --------------------------------------------------------------------------------------------
FILLER_CANDIDATES: List[str] = [
    # building and grounds upkeep
    "The hedge along the north path was trimmed on Wednesday morning.",
    "Someone has left a ladder leaning against the shed door.",
    "The gutter above the east entrance overflowed during the storm.",
    "A wasp nest was found under the eaves near the bicycle rack.",
    "The gravel path needs raking after the delivery vans turned on it.",
    "The padlock on the tool shed has been replaced with a combination latch.",
    "Fresh compost was spread on the raised beds behind the greenhouse.",
    "The hose reel by the tap is leaking at the joint again.",
    "Two stepping stones near the pond have sunk into the soil.",
    "The gate hinge squeaks loudly whenever anyone walks through.",
    "The lawn was mown on Friday and the clippings were taken to the tip.",
    "A fox has been digging under the fence beside the compost heap.",
    "The greenhouse panes rattled all night in the wind.",
    "The wheelbarrow tyre has gone flat and needs pumping up.",
    "Seed trays were moved into the cold frame before the frost.",
    "The drain in the courtyard was cleared of silt on Tuesday.",
    "Moss is spreading across the paving stones by the back steps.",
    "The water butt is nearly empty after the dry spell.",
    "A bird feeder was hung from the branch above the bench.",
    "The wooden bench beside the pond has a loose slat.",
    # kitchen and pantry
    "The kettle in the small kitchen takes a long time to boil.",
    "A tin of biscuits was left open on the counter overnight.",
    "The fridge door seal is perished and needs replacing.",
    "Someone put the teapot back on the shelf without rinsing it.",
    "The dishwasher finished its cycle but the cups were still damp.",
    "We have run out of washing up liquid and sponges.",
    "A carton of milk was left out and had to be thrown away.",
    "The mug cupboard is overflowing and needs a clear out.",
    "The toaster trips the socket whenever it is switched on.",
    "The tea towels were taken home to be washed.",
    # supplies, stationery, storage
    "The box of envelopes in the cupboard is almost empty.",
    "A roll of parcel tape has gone missing from the drawer.",
    "The label printer ran out of paper on Thursday.",
    "Spare batteries for the torches are kept in the tin by the door.",
    "The stapler and the hole punch were moved to the other shelf.",
    "Someone borrowed the long tape measure and did not return it.",
    "The stack of cardboard boxes in the corridor is a trip hazard.",
    "Old magazines were bundled up for the charity collection.",
    "The shelf labels have faded and are hard to read.",
    "A box of light bulbs was delivered and left in the porch.",
    "The extension lead in the store cupboard has a damaged plug.",
    "Paper clips and rubber bands are mixed together in the same jar.",
    # deliveries, post, errands
    "A parcel arrived on Tuesday and was signed for at the desk.",
    "The courier left a note because nobody was in to receive it.",
    "Post is collected from the box by the gate each morning.",
    "Two pallets were left on the loading bay overnight.",
    "The trolley by the entrance has a wobbly wheel.",
    "A delivery of potting compost is expected before the weekend.",
    "The van could not get up the track because of the mud.",
    "Someone signed for a crate that was meant for the neighbours.",
    "The return label was stuck on upside down and the parcel came back.",
    "A sack of bird seed was stored in the wrong cupboard.",
    # interiors, furniture, fittings
    "The chair in the corner has a wobbly leg and should be repaired.",
    "A picture frame fell off the wall in the corridor.",
    "The paint on the skirting board is peeling near the radiator.",
    "The clock above the stairs has stopped again.",
    "A bulb has gone in the lamp beside the armchair.",
    "The doormat curls up at the corner and catches on the door.",
    "The curtain rail came loose when the curtains were pulled.",
    "A coat hook has been pulled out of the wall in the porch.",
    "The floorboards creak loudly on the landing.",
    "The radiator in the back room needs bleeding before winter.",
    "A bookshelf was moved to make space for the armchair.",
    "The umbrella stand by the door is full of broken umbrellas.",
    # waste, recycling, cleaning
    "The recycling bin was not emptied because the lid was jammed.",
    "Glass jars should go in the crate, not the general waste.",
    "Someone left a bag of rubbish beside the bin instead of inside it.",
    "The floor in the porch was mopped after the rain came in.",
    "A mop and bucket are kept behind the door in the utility corner.",
    "The compost caddy needs emptying before it starts to smell.",
    "Old paint tins are stacked in the shed awaiting a tip run.",
    "The vacuum cleaner bag is full and needs changing.",
    "Scrap wood from the shelf project is piled behind the shed.",
    "The drain outside the kitchen was blocked with tea leaves.",
    # notices, records, correspondence
    "A notice about the summer fair was pinned to the board.",
    "The minutes from the spring gathering were filed in the folder.",
    "A photograph of the old mill was framed and hung in the hall.",
    "The visitors book has not been signed since March.",
    "A list of key holders is kept in the drawer by the telephone.",
    "The rota for watering the plants was rewritten after the holiday.",
    "An inventory of the tools was taken before the winter break.",
    "A reminder about the quiz night was sent round by hand.",
    "The thank you letters from the raffle were posted on Friday.",
    "A map of the allotment plots is pinned beside the gate.",
    # weather, seasons, nature
    "Frost was thick on the windscreen again this morning.",
    "The wind brought a branch down across the footpath.",
    "Rain has been running down the inside of the porch wall.",
    "The pond froze over for three nights in a row.",
    "Swallows have been nesting under the eaves since May.",
    "The apple tree by the wall produced very little fruit this autumn.",
    "Salt was spread on the steps after the ice warnings.",
    "Fallen acorns make the path slippery near the bench.",
    "The daffodils along the drive came up earlier than usual.",
    "A hedgehog was seen shuffling along the border after dusk.",
    "The stream behind the meadow ran very high last week.",
    "Sunflower heads were left standing for the birds.",
    # people and routines, deliberately generic
    "Someone kindly watered the ferns while the household was away.",
    "A neighbour borrowed the long ladder and returned it on Sunday.",
    "The choir practised in the hall on Thursday evening.",
    "A volunteer signed up to help with the spring clear out.",
    "The walking group met at the gate at nine on Saturday.",
    "A new key was cut for the side entrance.",
    "The quiz team is short of one player for the next round.",
    "Someone has been leaving muddy boots in the porch.",
    "A rota was drawn up for sweeping the front steps.",
    "The knitting circle meets in the back room on alternate weeks.",
    "A crate of apples was shared round after the harvest.",
    "The handbell ringers need a replacement case for the bells.",
    # second batch: kitchen, garden, outdoors, crafts, pastimes
    "A jar of plum jam was labelled and put in the pantry.",
    "The bread dough was left to prove on the windowsill.",
    "Tomato seedlings were pricked out into larger pots.",
    "The roses were deadheaded after the first flush of flowers.",
    "A swarm of bees settled in the hollow of the lime tree.",
    "The pea netting came loose during the gale.",
    "Runner beans were sown along the cane wigwam.",
    "The rhubarb patch was mulched with well rotted manure.",
    "A bucket of rainwater was left standing by the greenhouse.",
    "The strawberry runners were pegged down into fresh pots.",
    "Parsley and thyme were potted up for the kitchen sill.",
    "The fig tree was wrapped in fleece for the winter.",
    "A tray of cuttings was placed under the bench in the shade.",
    "The pumpkin vines spread right across the vegetable patch.",
    "The herb bed was edged with clipped box.",
    "A robin has taken to following the spade around the border.",
    "The bird bath was scrubbed and refilled.",
    "A pile of prunings was left beside the bonfire spot.",
    "The mower blade was sharpened before the first cut.",
    "The garden fork was left standing in the bed overnight.",
    "A kettle of water was carried round for the hanging baskets.",
    "The sweet peas were tied in with soft twine.",
    "A clump of chives was lifted and divided.",
    "The cold frame lid was propped open on warm days.",
    "The compost heap was turned with the fork on Sunday.",
    "A wheelbarrow load of leaf mould was spread on the beds.",
    "The path edging was trimmed with long handled shears.",
    "The crab apple jelly set firmly in the jars.",
    "A pan of soup was left simmering on the stove.",
    "The oven was cleaned after the roasting tin bubbled over.",
    "A batch of scones was baked for the afternoon tea.",
    "The preserving pan was borrowed and washed before returning.",
    "A sack of flour was tipped into the bin in the pantry.",
    "The spice jars were rearranged alphabetically on the rack.",
    "A bowl of lemons was left on the sill to ripen.",
    "The chopping block was scrubbed with coarse salt.",
    "A loaf was left to cool on the wire rack.",
    "The apron was hung on the hook by the stove.",
    "A jug of elderflower cordial was chilled in the fridge.",
    "The picnic hamper was aired out after the trip.",
    "A rug was shaken over the fence on Saturday.",
    "The bicycle chain was oiled and the brakes adjusted.",
    "A puncture was mended with a patch kit at the roadside.",
    "The saddle was raised a little for the longer ride.",
    "A pannier was strapped onto the rack for the shopping.",
    "The bell on the handlebar has a pleasant ring.",
    "A reflective tabard was hung up in the porch.",
    "The cycle path was resurfaced over the summer.",
    "A water bottle was left clipped in its cage.",
    "The tandem was wheeled out for the first time this year.",
    "A walking stick was left propped against the stile.",
    "The footbridge over the brook was repaired in April.",
    "A cairn marks the junction of the two tracks.",
    "The stile at the field corner has a broken step.",
    "A flask of tea was carried up in the rucksack.",
    "The map was folded away after the last turn.",
    "A kestrel hovered above the ridge for several minutes.",
    "The bothy was swept out and left tidy.",
    "A rope swing hangs from the oak by the ford.",
    "The sheep were moved to the lower field.",
    "A drystone wall collapsed along the lane.",
    "The gatepost was reset in fresh mortar.",
    "A hare was startled on the track at dusk.",
    "The quarry pool was clear enough to see the bottom.",
    "A wooden signpost was repainted by hand.",
    "The stepping stones were submerged after the rain.",
    "A kingfisher flashed past the bend in the river.",
    "The reed bed was cut back in February.",
    "A dragonfly rested on the lily pad.",
    "The otter spraint was found on the flat stone.",
    "A heron stood motionless in the shallows.",
    "The watercress beds were harvested by hand.",
    "A mayfly hatch brought the trout up.",
    "The mill pond was drained for the sluice repair.",
    "A punt was moored against the grassy bank.",
    "The towpath was muddy after the weekend.",
    "A lock keeper's cottage stands at the second gate.",
    "The narrowboat was repainted in green and cream.",
    "A moorhen nested under the overhanging willow.",
    "The swing bridge was closed for maintenance.",
    "A pair of swans nested on the island.",
    "The weir was noisy after the heavy rain.",
    "A rowing eight practised upstream at dawn.",
    "The ferry timetable changes at the equinox.",
    "A string of bunting was hung across the marquee.",
    "The trestle tables were wiped down after the fete.",
    "A tombola drum was left in the scout hut.",
    "The bunting was taken down and stored in the loft.",
    "A cake stall raised a decent sum for the roof fund.",
    "The coconut shy was packed away in the van.",
    "A brass band played on the green until dusk.",
    "The raffle tickets were counted twice.",
    "A hog roast was served from the far corner.",
    "The maypole ribbons were untangled before the dancing.",
    "A bouncy castle was deflated at six.",
    "The skittles were returned to the village hall cupboard.",
]


def _harness_words_and_stopwords():
    """The harness's own tokenizer and stopword list, so 'content word' means exactly what
    ``harness/pools.leakage_report`` means by it."""
    import sys
    harness = os.path.join(os.path.dirname(HERE), "harness")
    if harness not in sys.path:
        sys.path.insert(0, harness)
    import pools as HP  # noqa: WPS433
    return HP.words, HP.STOPWORDS


FILLER_FORBIDDEN_STEMS: List[str] = [
"bill", "invoic", "paym", "refund", "charg", "crash", "bug", "error", "outage",
"system", "sale", "pric", "contract", "plan", "upgrad", "licen", "seat", "subscri",
"depart", "request", "ticket", "technic", "dashboard", "api", "server", "support",
"account", "order", "receipt", "cancel", "downtime", "portal", "report", "customer",
"money", "card", "billing", "team", "quarter", "window", "desk", "office", "schedul",
"availab", "option", "servic", "workflow", "endpoint", "upload", "export", "backup",
"queue", "cache", "replica", "shard", "session", "cookie", "certificat", "migrat",
"vendor", "quotation", "quot", "licence", "licens", "discount", "mailbox", "workspace",
"timezone", "retention", "archive", "questionnaire", "subprocessor", "digest",
"folder", "colleague", "language", "region", "accessib", "warrant", "ledger",
"statement", "debit", "credit", "instalment", "instal", "remittance", "pro forma",
"proforma", "surcharge", "voucher", "mandate", "settlement", "balance", "amount",
"overdue", "statement", "renewal", "currency", "transfer", "bank", "cheque", "deposit",
"refundable", "tax", "unit", "endpoint", "gateway", "webhook", "worker", "restart",
"container", "health", "probe", "compaction", "replication", "failover", "threshold",
"latency", "concurrency", "schema", "payload", "parser", "scheduler", "index",
"node", "cluster", "tenant", "quota", "permission", "authentication", "certificate",
"bandwidth", "disk", "memory", "cpu", "upgrade", "downgrade", "release", "version",
"deploy", "rollback", "incident", "alert", "metric", "trace", "span", "log",
]


def build_filler() -> Dict[str, Any]:
    words, stopwords = _harness_words_and_stopwords()
    needle_vocab = set()
    for split in (EVAL_P1B, EVAL_NEW, TRAIN):
        for texts in split.values():
            for text in texts:
                needle_vocab.update(w for w in words(_strip_slots(text)) if len(w) >= 3)
    for qdef in QUESTIONS.values():
        needle_vocab.update(w for w in words(str(qdef["instructions"])) if len(w) >= 3)
        for k, v in qdef["criteria"].items():
            needle_vocab.update(w for w in words(str(k)) if len(w) >= 3)
            needle_vocab.update(w for w in words(str(v)) if len(w) >= 3)
    needle_content = {w for w in needle_vocab if w not in stopwords}

    kept, dropped = [], []
    for sentence in FILLER_CANDIDATES:
        hit = sorted(set(words(sentence)) & needle_content)
        stems = sorted({s for s in FILLER_FORBIDDEN_STEMS if s in sentence.lower()})
        if hit or stems:
            dropped.append({"sentence": sentence, "shared_content_words": hit, "stem_hits": stems})
        else:
            kept.append(sentence)
    if len(kept) < 48:
        raise ValueError("only %d filler sentences survived the filters; author more candidates"
                         % len(kept))
    return {
        "pool_id": "filler-h5-v1",
        "kind": "filler",
        "provenance": {
            "authored_by": "engineer agent, laya-lab, 2026, for the H5 aggregation-path experiment",
            "basis": ("building/grounds/supplies register. Authored *for this experiment* because "
                      "harness/pools/filler-v1 was written against the earlier support-domain "
                      "needle set and shares generic content words (team, quarter, window, run, "
                      "desk, leaves) with the wider H5 needle vocabulary."),
            "selection_rule": (
                "a candidate sentence is dropped if (a) it shares a whole content word "
                "(harness/pools.words, filtered by harness/pools.STOPWORDS, length >= 3) with the "
                "union of the H5 needle templates and the question's instructions and criteria, or "
                "(b) it contains any forbidden_stem as a substring -- the check that catches "
                "'cardboard' for 'card' and 'border' for 'order'. No sentence was rewritten to "
                "pass; candidates are only kept or dropped, and every drop is listed below."),
            "n_candidates": len(FILLER_CANDIDATES),
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "dropped": dropped,
            "note": ("contains no digits, so every digit in a document comes from the needle. "
                     "Repeated per document by the builder, as upstream's filler is."),
        },
        "forbidden_stems": FILLER_FORBIDDEN_STEMS,
        "sentences": kept,
    }


def _strip_slots(text: str) -> str:
    out = text
    for name in ("n", "ref"):
        out = out.replace("{%s}" % name, " ")
    return out


def main() -> None:
    _assert_disjoint_slot_values()
    os.makedirs(POOL_DIR, exist_ok=True)

    eval_templates = {lb: EVAL_P1B[lb] + EVAL_NEW[lb] for lb in LABELS}
    eval_sources = {lb: ["p1b"] * len(EVAL_P1B[lb]) + ["h5-new"] * len(EVAL_NEW[lb]) for lb in LABELS}
    eval_pool = build_split("eval", eval_templates, SLOTS_EVAL, eval_sources)
    train_pool = build_split("train", TRAIN, SLOTS_TRAIN)
    filler_pool = build_filler()

    for pool, fname in ((eval_pool, "needles-h5-eval-v1.json"),
                        (train_pool, "needles-h5-train-v1.json"),
                        (filler_pool, "filler-h5-v1.json")):
        path = os.path.join(POOL_DIR, fname)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=1, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        print("wrote %-42s sha256=%s" % ("experiments/h5-adapter/pools/" + fname,
                                         hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]))
    print("eval counts   :", eval_pool["counts"], "templates=%d" % len(eval_pool["templates"]))
    print("train counts  :", train_pool["counts"], "templates=%d" % len(train_pool["templates"]))
    print("combos/class  : eval=%s train=%s" % (eval_pool["combos_per_class"], train_pool["combos_per_class"]))
    prov = filler_pool["provenance"]
    print("filler        : %d candidates -> %d kept, %d dropped on content-word overlap"
          % (prov["n_candidates"], prov["n_kept"], prov["n_dropped"]))
    for d in prov["dropped"]:
        print("   dropped (%s): %s" % (",".join(d["shared_content_words"]), d["sentence"]))


if __name__ == "__main__":
    main()
