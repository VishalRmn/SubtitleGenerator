"""Handles formatting transcription results into subtitle files (SRT)."""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

from .models import TranscriptionResult, Segment
from .exceptions import FormattingError
from .utils import format_time_srt

logger = logging.getLogger(__name__)

class SubtitleFormatter(ABC):
    """Abstract base class for subtitle formatters."""

    @abstractmethod
    def format_subtitles(
        self,
        transcription_result: TranscriptionResult,
        output_path: str,
        max_chars_per_segment: int = 80,
        max_duration_seconds: float = 7.0,
        max_lines_per_block: int = 2
    ) -> None:
        """
        Formats the transcription result into a subtitle file.

        Args:
            transcription_result: The result from the transcription process.
            output_path: The path to save the formatted subtitle file.
            max_chars_per_segment: Max characters before trying to split a line/segment.
            max_duration_seconds: Max duration of a single subtitle block.
            max_lines_per_block: Max number of lines in a single subtitle block.

        Raises:
            FormattingError: If formatting or writing fails.
            FileSystemError: If the output directory is invalid.
        """
        pass


class SRTFormatter(SubtitleFormatter):
    """Formats subtitles into the SRT (SubRip Text) format."""

    def _split_segment(
        self,
        segment: Segment,
        max_chars: int,
        max_duration: float,
        max_lines: int
        ) -> List[Segment]:
        """
        Splits a single segment if it exceeds limits (basic implementation).

        This is a simple approach. More sophisticated methods would involve
        sentence boundary detection or word-level timing. This version splits
        based on character length primarily, and then duration.

        Args:
            segment: The input Segment object.
            max_chars: Maximum characters per new sub-segment (roughly per line * max_lines).
            max_duration: Maximum duration in seconds per new sub-segment.
            max_lines: Max lines (used implicitly by max_chars).

        Returns:
            A list of Segment objects, potentially containing more than one if split.
        """
        duration = segment.end_time - segment.start_time
        text = segment.text
        
        # Prioritize splitting by duration if it's significantly exceeded
        if duration > max_duration * 1.5: # Add some tolerance
             num_splits_duration = int(duration // max_duration) + 1
             split_duration = duration / num_splits_duration
             
             split_segments = []
             current_time = segment.start_time
             # Simple text split proportional to duration - less accurate
             words = text.split()
             words_per_split = len(words) // num_splits_duration if num_splits_duration > 0 else len(words)

             start_word_idx = 0
             for i in range(num_splits_duration):
                 end_word_idx = min(start_word_idx + words_per_split, len(words)) if i < num_splits_duration -1 else len(words)
                 sub_text = " ".join(words[start_word_idx:end_word_idx])
                 if not sub_text: continue

                 sub_start = current_time
                 sub_end = min(current_time + split_duration, segment.end_time)
                 # Avoid zero duration subs
                 if sub_end <= sub_start: sub_end = sub_start + 0.1 

                 split_segments.append(Segment(start_time=sub_start, end_time=sub_end, text=sub_text))
                 current_time = sub_end
                 start_word_idx = end_word_idx
                 
             if split_segments: 
                # Try to recursively split the resulting segments if they still violate char limits (less ideal)
                final_segments = []
                for seg in split_segments:
                    final_segments.extend(self._split_segment_by_chars(seg, max_chars, max_lines))
                logger.debug(f"Split segment by duration ({duration:.2f}s > {max_duration:.2f}s) into {len(final_segments)} parts.")
                return final_segments


        # If duration is acceptable, check character length
        if len(text) > max_chars:
             split_segs = self._split_segment_by_chars(segment, max_chars, max_lines)
             if len(split_segs) > 1:
                logger.debug(f"Split segment by chars ({len(text)} > {max_chars}) into {len(split_segs)} parts.")
             return split_segs

        # No split needed
        return [segment]

    def _split_segment_by_chars(self, segment: Segment, max_chars: int, max_lines: int) -> List[Segment]:
        """Helper to split primarily by character count, distributing time."""
        text = segment.text
        if len(text) <= max_chars:
            return [segment]

        words = text.split()
        if not words:
            return [segment] # Should not happen if text exists

        split_segments = []
        current_line = ""
        current_char_count = 0
        start_word_idx = 0
        
        num_desired_splits = (len(text) // max_chars) + 1
        duration = segment.end_time - segment.start_time
        time_per_char = duration / len(text) if len(text) > 0 else 0

        current_start_time = segment.start_time
        processed_chars = 0

        lines_in_block = []

        for i, word in enumerate(words):
            word_len = len(word)
            potential_line = f"{current_line} {word}" if current_line else word
            
            # Check if adding the word exceeds max_chars for the *entire block*
            if len(potential_line) > max_chars:
                # Finish the current segment block
                if current_line:
                    lines_in_block.append(current_line)
                    block_text = "\n".join(lines_in_block)
                    block_chars = len(block_text) - (len(lines_in_block)-1) # Exclude newline chars approx
                    
                    sub_end_time = current_start_time + (processed_chars * time_per_char)
                    # Ensure end time doesn't exceed original segment end time
                    sub_end_time = min(sub_end_time, segment.end_time)
                    # Prevent zero or negative duration
                    if sub_end_time <= current_start_time: sub_end_time = current_start_time + 0.1

                    split_segments.append(Segment(start_time=current_start_time, end_time=sub_end_time, text=block_text))
                    
                    current_start_time = sub_end_time # Start next segment from end of last one
                    processed_chars = 0 # Reset char count for time calculation
                    lines_in_block = []
                    current_line = word # Start new block/line with current word
                else: # Word itself is longer than max_chars, keep it together
                     current_line = word

                # If the new current_line also triggers a new block immediately, handle it
                if len(current_line) > max_chars // max_lines: # Check approx line length
                     lines_in_block.append(current_line) # Add this long line
                     # If block is full or it's the last word, create segment now
                     if len(lines_in_block) >= max_lines or i == len(words) - 1:
                         block_text = "\n".join(lines_in_block)
                         block_chars = len(block_text) - (len(lines_in_block)-1)
                         processed_chars += block_chars # Update processed chars for timing

                         sub_end_time = current_start_time + (processed_chars * time_per_char)
                         sub_end_time = min(sub_end_time, segment.end_time)
                         if sub_end_time <= current_start_time: sub_end_time = current_start_time + 0.1

                         split_segments.append(Segment(start_time=current_start_time, end_time=sub_end_time, text=block_text))

                         current_start_time = sub_end_time
                         processed_chars = 0
                         lines_in_block = []
                         current_line = "" # Reset line since block was formed
                     # else: line added, continue to next word

                else: # Word starts a new line within the current block
                     lines_in_block.append(current_line)
                     processed_chars += len(current_line) +1 # Add chars + space/newline approx
                     current_line = "" # Reset line, wait for next word

            else: # Word fits
                current_line = potential_line
                # If it fits AND makes the line longer than ideal single line length, add newline potential
                if len(current_line) > max_chars // max_lines and len(lines_in_block) < max_lines -1 :
                     lines_in_block.append(current_line)
                     processed_chars += len(current_line) + 1
                     current_line = "" # Ready for next line in block
                # else: word added to current line, continue

        # Add any remaining text as the last segment block
        if current_line:
            lines_in_block.append(current_line)

        if lines_in_block:
            block_text = "\n".join(lines_in_block)
            # Use remaining duration for the last block
            split_segments.append(Segment(start_time=current_start_time, end_time=segment.end_time, text=block_text))

        # Filter out empty segments potentially created during split logic
        return [seg for seg in split_segments if seg.text and seg.end_time > seg.start_time]


    def format_subtitles(
        self,
        transcription_result: TranscriptionResult,
        output_path: str,
        max_chars_per_segment: int = 80,
        max_duration_seconds: float = 7.0,
        max_lines_per_block: int = 2
    ) -> None:
        """
        Formats transcription segments into an SRT file with segmentation.

        Args:
            transcription_result: The result containing segments.
            output_path: Path to save the SRT file.
            max_chars_per_segment: Max characters per block before splitting.
            max_duration_seconds: Max duration per block before splitting.
            max_lines_per_block: Max lines per block (used in splitting logic).

        Raises:
            FormattingError: If file writing fails.
            FileSystemError: If output directory is invalid (checked implicitly by open).
        """
        logger.info(f"Formatting subtitles to SRT: {output_path}")
        logger.info(f"Segmentation rules: max_chars={max_chars_per_segment}, max_duration={max_duration_seconds}s, max_lines={max_lines_per_block}")

        subtitle_index = 1
        formatted_segments = []

        max_chars_combined = max_chars_per_segment * max_lines_per_block # Rough total chars for block

        for segment in transcription_result.segments:
            # Apply splitting logic
            split_segments = self._split_segment(
                segment,
                max_chars=max_chars_combined,
                max_duration=max_duration_seconds,
                max_lines=max_lines_per_block
            )
            formatted_segments.extend(split_segments)


        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in formatted_segments:
                     # Final check for line breaks within the text based on max_chars_per_line
                    lines = segment.text.split('\n')
                    final_lines = []
                    for line in lines:
                         if len(line) > max_chars_per_segment:
                             # Basic word wrap for lines that are too long
                            wrapped_line = ""
                            line_len = 0
                            for word in line.split():
                                if line_len == 0:
                                    wrapped_line += word
                                    line_len += len(word)
                                elif line_len + len(word) + 1 <= max_chars_per_segment:
                                    wrapped_line += f" {word}"
                                    line_len += len(word) + 1
                                else:
                                    final_lines.append(wrapped_line)
                                    wrapped_line = word
                                    line_len = len(word)
                            final_lines.append(wrapped_line) # Add last part
                         else:
                             final_lines.append(line)

                    # Limit to max_lines_per_block
                    final_text = "\n".join(final_lines[:max_lines_per_block])


                    start_time_str = format_time_srt(segment.start_time)
                    end_time_str = format_time_srt(segment.end_time)

                    # Ensure end time is strictly after start time
                    if segment.end_time <= segment.start_time:
                        logger.warning(f"Segment {subtitle_index} has zero or negative duration ({start_time_str} -> {end_time_str}). Adjusting end time slightly.")
                        segment.end_time = segment.start_time + 0.1 # Add small duration
                        end_time_str = format_time_srt(segment.end_time)


                    f.write(f"{subtitle_index}\n")
                    f.write(f"{start_time_str} --> {end_time_str}\n")
                    f.write(f"{final_text}\n\n")
                    subtitle_index += 1

            logger.info(f"Successfully wrote {subtitle_index - 1} subtitle blocks to {output_path}")

        except IOError as e:
            logger.error(f"Failed to write SRT file to {output_path}: {e}", exc_info=True)
            raise FormattingError(f"Could not write SRT file: {e}") from e
        except Exception as e:
             logger.error(f"An unexpected error occurred during SRT formatting: {e}", exc_info=True)
             raise FormattingError(f"Unexpected formatting error: {e}") from e

# class VTTFormatter(SubtitleFormatter):
#     """Formats subtitles into the VTT (Web Video Text Tracks) format."""
#     def format_subtitles(self, transcription_result: TranscriptionResult, output_path: str, **kwargs) -> None:
#         logger.info(f"Formatting subtitles to VTT: {output_path}")
#         # Implementation similar to SRTFormatter, but uses VTT time format (HH:MM:SS.ms)
#         # and header ("WEBVTT\n\n")
#         # Time format uses '.' instead of ',' for milliseconds.
#         raise NotImplementedError("VTT formatting is not yet implemented.")