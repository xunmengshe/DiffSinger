# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## Customizing your phoneme system for DiffSinger*

*Exclusive in this forked repository.

### 0. Requirements

#### Limitations for dictionaries

The current code implementation supports customized grapheme-to-phoneme dictionaries, with the following limitations:

- The dictionary must be a two-part phoneme system. Namely, one syllable should contain at most two phones, where only two cases are allowed: 1. one consonant + one vowel, 2. one single vowel.
- `AP` (aspiration) and `SP` (space) will be included in the phoneme list and cannot be removed.

#### Requirements for data labels

The preprocessing schedule introduced validations for the data labels to avoid mismatch between phoneme labels and the dictionary. Thus, your data label must meet the following requirements:

- The data must contain labels for `AP` and `SP` This is due to some code implementation issues, and may be fixed in the future.
- Tha data labels must, and must only contain all phonemes that appear in the dictionary. Otherwise, the coverage and oov checks will fail.

### 1. Preparation

A dictionary file is required to use your own phoneme system. A dictionary is a syllable-to-phoneme table, with each line in the following form (`\t` to separate the syllable and its phonemes):

```
<syllable>	[<consonant>] <vowel>
```

For example, the following rules are valid in a dictionary file:

```
a	a
ai	ai
ang	ang
ba	b a
bai	b ai
ban	b an
```

Note that you do not need (actually you must not) put `AP` and `SP` in the dictionary.

If one syllable is mapped to one phoneme, the phoneme will be regarded as a vowel. If one syllable is mapped to two phonemes, the first one will be regarded as a consonant and the other as a vowel. Syllables that are mapped to more than two phonemes are not allowed.

Vowel phonemes are used to align with the head of the note to keep the song in a correct rhythm. See this [issue](https://github.com/MoonInTheRiver/DiffSinger/issues/60) for explanations.

It is reasonable for the dictionary to design a unique symbol for each pronunciation. If one symbol have multiple pronunciations based on different context, especially when one pronunciation has many occurences while the others have only a few, the network may not learn very well of the rules, leading to more pronunciation errors at inference time.

### 2. Preprocessing and inference

#### Configurations

To preprocess your data with a customized dictionary, you should specify the dictionary path in the config file:

```yaml
g2p_dictionary: path/to/your/dictionary.txt
```

If not specified, this hyperparamerter will fall back to `dictionaries/opencpop.txt` for backward compatibility.

#### Phoneme distribution summary

When preprocessing, the program will generate a summary of the phoneme occurrence distribution summary. The summary includes messages in the standard output and a JPG file in the preprocessing directory. The summary only covers phonemes in the dictionary, along with `AP` and `SP`. Try to balance number of occurrences of each phoneme for more stable pronunciations at inference time.

#### Coverage and OOV checks

The program will perform phoneme coverage checks and OOV detection based on the given dictionary and data label. These checks fail when:

- Some phonemes in the dictionary have not appeared in the data labels (not a full coverage of the dictionary)
- Some phonemes are not in the dictionary but appear in the data labels (unrecognized symbols)

The program will throw an `AssertionError` and show differences of the dictionary phoneme set and the actual data phoneme set like below:

```
AssertionError: transcriptions and dictionary mismatch.
 (+) ['E', 'En', 'i0', 'ir']
 (-) ['AP', 'SP']
```

This means there are 4 unexpected symbols in the data labels (`ir`, `i0`, `E`, `En`) and 2 missing phonemes that are not covered by the data labels (`AP`, `SP`).

#### Inference with a custom dictionary

When doing inference, the program will read the dictionary file from the checkpoint folder and generate a phoneme set. There are two ways of inference:

- Inference with automatic phoneme durations, i.e. inputting syllables and matching the left row of the dictionary. Each vowel is aligned with the head of the note, and consonants have their duration predicted by the network.
- Inference with manual phoneme durations, i.e. directly inputting phoneme-level durations. Every phoneme should be in the phoneme set.

### 3. Preset dictionaries

There are currently two preset dictionaries.

#### The original Opencpop dictionary [[source]](../dictionaries/opencpop.txt)

The original Opencpop dictionary, which you can find [here](http://wenet.org.cn/opencpop/resources/annotationformat/), are fully aligned with the standard pinyin format of Mandarin Chinese. We copied the dictionary from the website, removed 5 syllables that has no occurrence in the data labels (`hm`, `hng`, `m`, `n` and `ng`) and added some aliases for some syllables (e.g. `jv` for `ju`). It has the most compatibility with the previous model weights, but may cause bad cases in pronunciations, especially in cases that the note is a slur. Thus, this dictionary is deprecated by default and remained only for backward compatibility.

Phoneme distribution of Opencpop dataset on this dictionary can be found [here](http://wenet.org.cn/opencpop/resources/statisticalinformation/).

#### The new strict pinyin dictionary [[source]](../dictionaries/opencpop-strict.txt)

We distinguished some different pronunciations of some phonemes, and added 4 phonemes to the original dictionary: `ir`, `i0`, `E` and `En`.

Some mapping rules are changed:

- `zhi`, `chi`, `shi`, `ri` are mapped to `zh ir`, `ch ir`, `sh ir`, `r ir` (distinguished from orignal `i`)
- `zi`, `ci`, `si` are mapped to `z i0`, `c i0`, `s i0` (distinguished from original `i`)
- `ye` are mapped to `y E` (distinguished from original `e`)
- `yan` are mapped to `y En` (distinguished from original `an`)

Phoneme distribution* of Opencpop dataset on this dictionary is shown below.

![img](resources/phoneme_distribution.jpg)

*`AP` and `SP` are not included.

To migrate `ds` file from original dictionary to this strict dictionary, run the following command:

```bash
python utils/phoneme_utils path/to/your/original.ds path/to/your/target.ds
```

