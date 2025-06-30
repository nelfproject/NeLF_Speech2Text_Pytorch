"""
Scripts to generate segments from VAD outputs that are suitable for speech decoding.
"""

def sequential_merge_vad_segments(segments, max_pause=1.0, max_segment_length=30.0):
    if not segments:
        return []

    merged_segments = []
    current_start = segments[0]['start']
    current_end = segments[0]['end']

    for i in range(1, len(segments)):
        next_start = segments[i]['start']
        next_end = segments[i]['end']
        pause = next_start - current_end
        potential_duration = next_end - current_start

        if pause <= max_pause and potential_duration <= max_segment_length:
            # Merge the segments
            current_end = next_end
        else:
            # Finalize current segment
            merged_segments.append({'start': current_start, 'end': current_end})
            # Start new segment
            current_start = next_start
            current_end = next_end

    # Add last segment
    merged_segments.append({'start': current_start, 'end': current_end})
    return merged_segments

def hierarchical_merge_vad_segments(segments, max_pause=2.0, max_segment_length=30.0, min_segment_length=0.0):
    if not segments:
        return []

    # Work on a copy to avoid modifying original list
    merged = segments[:]

    def compute_pause(i):
        return merged[i+1]['start'] - merged[i]['end']

    while True:
        # Find all mergeable pairs
        candidate_pairs = []
        for i in range(len(merged) - 1):
            pause = compute_pause(i)
            new_start = merged[i]['start']
            new_end = merged[i+1]['end']
            duration = new_end - new_start
            if pause <= max_pause and duration <= max_segment_length:
                candidate_pairs.append((pause, i, duration))

        if not candidate_pairs:
            break

        # Choose the pair with the smallest pause (greedy)
        _, best_idx, _ = min(candidate_pairs, key=lambda x: x[0])

        # Merge the two segments
        merged_segment = {
            'start': merged[best_idx]['start'],
            'end': merged[best_idx + 1]['end']
        }

        # Replace the two segments with their merged version
        merged = merged[:best_idx] + [merged_segment] + merged[best_idx + 2:]

    if min_segment_length == 0.0:
        return merged

    # Phase 2: Ensure minimum segment length
    i = 0
    while i < len(merged):
        duration = merged[i]['end'] - merged[i]['start']
        if duration < min_segment_length:
            # Try to merge with the shorter adjacent segment
            can_merge_prev = i > 0
            can_merge_next = i < len(merged) - 1

            best_choice = None
            if can_merge_prev:
                prev_dur = merged[i]['end'] - merged[i - 1]['start']
                if prev_dur <= max_segment_length:
                    best_choice = 'prev'
            if can_merge_next:
                next_dur = merged[i + 1]['end'] - merged[i]['start']
                if next_dur <= max_segment_length:
                    if best_choice != 'prev' or next_dur < prev_dur:
                        best_choice = 'next'

            if best_choice == 'prev':
                merged[i - 1] = {
                    'start': merged[i - 1]['start'],
                    'end': merged[i]['end']
                }
                merged.pop(i)
                i -= 1  # stay at merged index
            elif best_choice == 'next':
                merged[i] = {
                    'start': merged[i]['start'],
                    'end': merged[i + 1]['end']
                }
                merged.pop(i + 1)
            else:
                # Cannot fix short segment
                i += 1
        else:
            i += 1

    return merged

