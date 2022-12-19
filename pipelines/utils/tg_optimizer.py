import parselmouth as pm
import textgrid as tg


def optimize_textgrid(textgrid: tg.TextGrid, sound: pm.Sound):
    sr = sound.sampling_frequency
    f0 = sound.to_pitch_ac(
        time_step=0.005,
        voicing_threshold=0.6,
        pitch_floor=40.,
        pitch_ceiling=1100.).selected_array['frequency']

