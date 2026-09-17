"""Paraphrase sets for the text-conditioning path.

With one caption per mood, the text path hands the trainable projection
exactly two points, so it can get away with learning two addresses instead of
the happy<->sad *direction*. Sampling a different paraphrase each training
step gives each mood a distribution with real spread, and forces the
projection to respond to what the paraphrases share (valence) rather than to
one particular CLAP vector.

Design rules for these lists:
  * Valence-coded, not arousal-coded. Happy spans calm-happy ("peaceful,
    content") through energetic-happy ("euphoric dance"), matching how the
    labels are derived (annotations.mood_from_va ignores arousal).
  * Instruments and genres are mirrored across the two moods (happy piano /
    sad piano, joyful choir / mournful choir, ...). If "cello" appeared only on
    the sad side, the projection could learn "cello => sad" — a timbre cue, not
    a mood cue.
  * Every paraphrase is checked against CLAP itself: its text embedding must sit
    closer to its own mood's canonical caption than to the other's. CLAP's
    "happy" caption leans upbeat, so gentle wordings ("serene and blissful",
    "calm, peaceful and content") land on the SAD side and would train the text
    path on vectors the judge reads as the opposite mood. Those were reworded
    with explicit valence words ("a happy, smiling and peaceful acoustic song")
    rather than dropped, so calm-happy music stays represented. Re-run the check
    after editing these lists.
  * The canonical caption from annotations.MOOD_PROMPTS is always included, so
    the exact string inference and evaluation use is trained on directly.

Only training samples from these. Inference and evaluation keep using the
single canonical caption per mood (annotations.MOOD_PROMPTS), so the judge is
unchanged and results stay comparable with earlier runs.
"""

from annotations import MOOD_HAPPY, MOOD_SAD, MOOD_PROMPTS

_HAPPY = [
    "a cheerful upbeat tune",
    "bright optimistic music",
    "a joyful song full of energy",
    "a feel-good pop song with a catchy melody",
    "cheerful, sunny acoustic guitar music",
    "a playful bouncy piano melody",
    "a triumphant orchestral piece",
    "a happy, upbeat whistling tune",
    "an exuberant celebration song",
    "a carefree summer anthem",
    "cheerful, upbeat strings",
    "a gleeful jazzy swing tune",
    "a cheerful folk song with ukulele",
    "euphoric dance music",
    "a cheerful, contented piano piece",
    "music that sounds happy and carefree",
    "a joyful, upbeat gospel choir",
    "a lively major-key melody",
    "a delightful cheerful waltz",
    "a sunny reggae groove",
    "a cheerful feel-good ballad",
    "an upbeat rock song that makes you want to smile",
    "bright cheerful synth pop",
    "a bubbly happy electronic track",
    "a festive, joyful brass band",
    "a hopeful, inspiring cinematic score",
    "a jubilant, upbeat piano piece",
    "a positive, energetic funk groove",
    "a happy, uplifting ambient track",
    "a happy, smiling and peaceful acoustic song",
    "a cheerful, upbeat love song",
    "an elated violin melody",
    "a merry country tune with fiddle",
    "a victorious fanfare",
    "an upbeat, cheerful morning song",
    "a relaxed happy bossa nova",
    "a sparkling optimistic soundtrack",
    "a cheerful, lighthearted indie pop song",
    "a buoyant, upbeat jazz piece",
    "a happy, uplifting chorus",
    "a happy, cheerful cello melody",
    "an upbeat, feel-good soul song",
    "a bright major-key orchestra",
    "a fun party song",
    "a peaceful, joyful flute melody",
    "upbeat music that makes you feel happy",
    "a happy, hopeful sunrise song",
    "an upbeat, cheerful ska tune",
    "a charming, happy harp melody",
    "a celebratory, joyful hip hop beat",
    "an optimistic, bright electronic anthem",
    "a light, happy acoustic strum",
    "a playful, happy clarinet tune",
    "a bright, uplifting worship song",
    "an upbeat, joyful choir",
    "an inspiring, feel-good rock anthem",
    "a dreamy and happy synth piece",
    "a happy trumpet solo",
    "a sunny, joyful string quartet",
]

_SAD = [
    "a somber downcast piano",
    "melancholy strings",
    "a heartbroken ballad",
    "a gloomy, sorrowful song",
    "a mournful solo cello",
    "slow, tearful acoustic guitar music",
    "a tragic orchestral piece",
    "a lonely, desolate melody",
    "a grieving funeral dirge",
    "a wistful, nostalgic song about loss",
    "bleak, depressing ambient music",
    "a sorrowful minor-key violin melody",
    "a dejected, weary blues song",
    "a bittersweet, regretful piano piece",
    "music that sounds sad and hopeless",
    "a mournful gospel hymn",
    "a despairing, anguished rock song",
    "a lamenting choir",
    "a heavy-hearted slow waltz",
    "a sad, lonely jazz ballad",
    "a tearful folk song",
    "a sorrowful synth pop song",
    "a desolate, sad electronic track",
    "a gloomy brass requiem",
    "a grim, hopeless cinematic score",
    "a melancholic harp melody",
    "an unhappy, brooding cello piece",
    "a sad country song with slide guitar",
    "a dreary, rainy-day piano piece",
    "a sad love song about heartbreak",
    "a miserable, pained vocal",
    "a mournful flute melody",
    "a sad, somber emo song",
    "an aching, sorrowful soul ballad",
    "a sad goodbye song",
    "a mournful, sad trumpet solo",
    "a depressing, dark ambient drone",
    "a sad lullaby",
    "a regretful, melancholy indie song",
    "a crushing, tragic symphony",
    "a despondent, slow hip hop beat",
    "a lonely night-time saxophone",
    "music full of grief and sorrow",
    "a sad, minor-key orchestra",
    "a heartbreaking string quartet",
    "a pensive, sorrowful acoustic song",
    "a gloomy, downbeat trip hop track",
    "a mournful clarinet tune",
    "a sad, empty and hollow soundscape",
    "a weeping violin",
    "a gloomy post-rock piece full of sadness",
    "a wistful, sad bossa nova",
    "a mournful choir singing",
    "a sad piano elegy",
    "a bleak, sorrowful organ piece",
    "music that sounds lonely and heartbroken",
    "a sad and tearful cello melody",
    "a melancholy, sad synth piece",
    "a depressing, gloomy rock ballad",
]

# Canonical caption first, so index 0 of each mood is the inference string.
MOOD_PARAPHRASES = {
    MOOD_HAPPY: [MOOD_PROMPTS[MOOD_HAPPY]] + _HAPPY,
    MOOD_SAD: [MOOD_PROMPTS[MOOD_SAD]] + _SAD,
}


def paraphrases_for(mood: str, enabled: bool = True) -> list:
    """Training captions for a mood.

    Falls back to the single canonical caption when paraphrasing is disabled
    or the mood has no paraphrase set (e.g. a heuristic-fallback label), which
    reproduces the old one-caption-per-mood behaviour exactly.
    """
    from annotations import mood_prompt
    if enabled and mood in MOOD_PARAPHRASES:
        return list(MOOD_PARAPHRASES[mood])
    return [mood_prompt(mood)]
