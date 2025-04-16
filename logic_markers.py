#!/usr/bin/env python3
"""
Logic Pro Marker Reader and Writer (Refactored)

A tool for reading and writing Logic Pro marker data in WAV audio files.
This version incorporates refactoring based on analysis, including:
- A dedicated `Marker` data class.
- Simplified data handling using `List[Marker]`.
- Removal of potentially ineffective metadata reading code for broader types.
- Consistent file handling in the writer.

Usage:
    # Display help and usage information
    python logic_markers.py --help

    # Read markers from an audio file
    python logic_markers.py read input.wav --csv_output markers.csv

    # Write markers to an audio file
    python logic_markers.py write --input_audio input.wav --csv_file markers.csv --output_audio output.wav

    # Write markers to a silent audio file
    python logic_markers.py create_from_csv --csv_file your_markers.csv --output_audio silent_with_markers.wav

    # Test the marker writing functionality
    python logic_markers.py test_write --source_audio source.wav --target_audio target.wav
"""
import os
import sys
import re
import struct
import fire
import csv
import shutil
import dataclasses
from typing import List, Dict, Optional, Union, Tuple, Generator, Any

# --- Constants ---
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_SMPTE_FPS = 30
DEFAULT_SMPTE_SUBFRAMES_PER_FRAME = 100

# --- Marker Data Structure ---
@dataclasses.dataclass(frozen=True)
class Marker:
    """Represents a single marker with ID, position, and label."""
    id: int
    position: int  # Always stored in samples
    label: str

    def position_smpte(self, sample_rate: int = DEFAULT_SAMPLE_RATE, fps: int = DEFAULT_SMPTE_FPS, subframes_per_frame: int = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME) -> str:
        """Convert sample position to SMPTE timecode."""
        total_seconds = self.position / sample_rate
        hours = int(total_seconds // 3600)
        total_seconds %= 3600
        minutes = int(total_seconds // 60)
        total_seconds %= 60
        seconds = int(total_seconds)
        fractional_seconds = total_seconds - seconds
        
        total_subframes = fractional_seconds * fps * subframes_per_frame
        frames = int(total_subframes // subframes_per_frame)
        subframes = int(round(total_subframes % subframes_per_frame)) # Round subframes

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.{subframes:02d}"

    @classmethod
    def from_smpte(cls, marker_id: int, smpte_time: str, label: str, sample_rate: int = DEFAULT_SAMPLE_RATE, fps: int = DEFAULT_SMPTE_FPS, subframes_per_frame: int = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME) -> 'Marker':
        """Create a Marker instance from SMPTE timecode."""
        match = re.match(r'(\d+):(\d+):(\d+):(\d+)(?:\.(\d+))?', smpte_time)
        if not match:
            raise ValueError(f"Invalid SMPTE timecode format: {smpte_time}")

        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        frames = int(match.group(4))
        subframes = int(match.group(5) or 0)

        total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / fps) + (subframes / (fps * subframes_per_frame))
        position = int(total_seconds * sample_rate)
        return cls(id=marker_id, position=position, label=label)

# --- Base Class ---
class LogicMarkerBase:
    """Base class for Logic Pro marker handling with shared functionality."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Store config if needed later, e.g., for different sample rates
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.smpte_fps = DEFAULT_SMPTE_FPS
        self.smpte_subframes = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME
            
    def log(self, message: str):
        """Print debug messages when verbose mode is enabled."""
        if self.verbose:
            print(f"{message}")

    def _smpte_to_samples(self, smpte_time: str) -> int:
        """Helper to convert SMPTE timecode to sample position using instance settings."""
        marker = Marker.from_smpte(0, smpte_time, "", self.sample_rate, self.smpte_fps, self.smpte_subframes)
        return marker.position

    def _samples_to_smpte(self, position: int) -> str:
        """Helper to convert sample position to SMPTE timecode using instance settings."""
        marker = Marker(id=0, position=position, label="")
        return marker.position_smpte(self.sample_rate, self.smpte_fps, self.smpte_subframes)

    def _read_wav_chunks(self, file_path: str) -> Generator[Tuple[bytes, bytes], None, None]:
        """Generator to read chunks from a WAV file."""
        try:
            with open(file_path, 'rb') as f_read:
                # Check RIFF header
                riff_id = f_read.read(4)
                if riff_id != b'RIFF':
                    raise ValueError("Not a valid WAV file: missing RIFF header")
                _ = f_read.read(4) # Skip file size
                wave_id = f_read.read(4)
                if wave_id != b'WAVE':
                    raise ValueError("Not a valid WAV file: missing WAVE format")

                self.log(f"[LogicMarkerBase._read_wav_chunks] Reading chunks from {file_path}...")
                while True:
                    chunk_header = f_read.read(8)
                    if len(chunk_header) < 8:
                        if len(chunk_header) > 0:
                            self.log(f"[LogicMarkerBase._read_wav_chunks] Warning: Trailing {len(chunk_header)} bytes found at end of file.")
                        break # End of file
                    
                    chunk_id = chunk_header[0:4]
                    chunk_size = int.from_bytes(chunk_header[4:8], byteorder='little')
                    chunk_data = f_read.read(chunk_size)

                    if len(chunk_data) < chunk_size:
                        self.log(f"[LogicMarkerBase._read_wav_chunks] Warning: Chunk {chunk_id!r} truncated. Expected {chunk_size}, got {len(chunk_data)}.")
                        # Continue with truncated data for now

                    yield chunk_id, chunk_data
                    
                    # Skip padding byte if chunk size was odd
                    if chunk_size % 2 != 0:
                        padding = f_read.read(1)
                        if not padding:
                            self.log(f"[LogicMarkerBase._read_wav_chunks] Warning: Expected padding byte after chunk {chunk_id!r} but reached EOF.")
                            break # Reached end unexpectedly
        except FileNotFoundError:
            raise
        except Exception as e:
            self.log(f"[LogicMarkerBase._read_wav_chunks] Error reading chunks: {e}")
            raise # Re-raise error

# --- Marker Reader ---
class LogicMarkerReader(LogicMarkerBase):
    """Class to read marker information from WAV audio files."""
    
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def read_markers_from_file(self, file_path: str) -> List[Marker]:
        """
        Read marker information from a WAV audio file.
        
        Args:
            file_path (str): Path to the audio file.
            
        Returns:
            List[Marker]: List of markers found, sorted by position.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[LogicMarkerReader.read_markers_from_file] File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext != '.wav':
            self.log(f"[LogicMarkerReader.read_markers_from_file] Warning: Only WAV file analysis is fully supported. File: {file_path}")
            # Attempt basic WAV read anyway, might fail gracefully or succeed if it's WAV-like
        
        self.log(f"[LogicMarkerReader.read_markers_from_file] Reading markers from WAV: {file_path}")
        markers = self._read_markers_from_wav_chunks(file_path)
        
        if not markers:
             self.log("[LogicMarkerReader.read_markers_from_file] No markers found in file")
        
        # Sort by position before returning
        markers.sort(key=lambda m: m.position)
        return markers
        
    def _read_markers_from_wav_chunks(self, file_path: str) -> List[Marker]:
        """Reads markers by processing WAV chunks."""
        marker_positions: Dict[int, int] = {} # cue_id -> position
        marker_labels: Dict[int, str] = {} # cue_id -> label
        bwf_data: Dict[str, Any] = {} # Store BWF data if needed

        try:
            for chunk_id, chunk_data in self._read_wav_chunks(file_path):
                chunk_name = chunk_id.decode('utf-8', errors='replace')
                self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Processing chunk: {chunk_name}, size: {len(chunk_data)}")

                if chunk_id == b'cue ':
                    positions = self._process_cue_chunk_data(chunk_data)
                    marker_positions.update(positions)
                elif chunk_id == b'LIST':
                    if chunk_data.startswith(b'adtl'):
                        labels = self._process_adtl_chunk_data(chunk_data[4:]) # Skip 'adtl' type ID
                        marker_labels.update(labels)
                    else:
                        self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Skipping non-adtl LIST chunk type: {chunk_data[:4]!r}")
                elif chunk_id == b'bext':
                    bwf = self._process_bext_chunk_data(chunk_data)
                    bwf_data.update(bwf) # Store BWF info if needed for context
                elif chunk_name.startswith(('APPL', 'appl', 'Logi')):
                     self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Found potential Logic chunk: {chunk_name}. Storing raw data.")
                     bwf_data[f"{chunk_name}_data"] = chunk_data # Store other potentially relevant chunks
                # else: skip other chunks

        except ValueError as e: # Catch WAV format errors from _read_wav_chunks
             self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Error reading WAV structure: {e}")
             return [] # Return empty list if basic WAV structure is invalid
        except Exception as e:
             self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Unexpected error processing chunks: {e}")
             return [] # Return empty on other errors during processing


        # Combine positions and labels into Marker objects
        markers = []
        for cue_id, position in marker_positions.items():
            label = marker_labels.get(cue_id, f"Marker {cue_id}") # Default label if missing
            markers.append(Marker(id=cue_id, position=position, label=label))
            
        self.log(f"[LogicMarkerReader._read_markers_from_wav_chunks] Found {len(markers)} markers.")
        return markers

    def _process_cue_chunk_data(self, chunk_data: bytes) -> Dict[int, int]:
        """Process cue chunk data to extract marker positions."""
        positions = {}
        try:
            num_cue_points = int.from_bytes(chunk_data[0:4], byteorder='little')
            self.log(f"[LogicMarkerReader._process_cue_chunk_data] Number of cue points: {num_cue_points}")
            
            offset = 4
            for i in range(num_cue_points):
                cue_point_data = chunk_data[offset : offset + 24]
                if len(cue_point_data) < 24:
                    self.log(f"[LogicMarkerReader._process_cue_chunk_data] Warning: Incomplete cue point data at index {i}.")
                    break
                    
                cue_id = int.from_bytes(cue_point_data[0:4], byteorder='little')
                # WAV spec cue point structure: id(4), pos(4), fccChunk(4), chunkStart(4), blockStart(4), sampleOffset(4)
                position = int.from_bytes(cue_point_data[20:24], byteorder='little') 
                
                positions[cue_id] = position
                self.log(f"[LogicMarkerReader._process_cue_chunk_data] Cue point {cue_id} at sample {position} ({self._samples_to_smpte(position)})")
                offset += 24
                
        except Exception as e:
            self.log(f"[LogicMarkerReader._process_cue_chunk_data] Error processing cue chunk data: {e}")
            
        return positions
        
    def _process_adtl_chunk_data(self, chunk_data: bytes) -> Dict[int, str]:
        """Process adtl chunk data for marker labels."""
        labels = {}
        offset = 0
        end_position = len(chunk_data)
        self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Processing adtl data (size: {end_position}).")
        
        try:
            while offset < end_position:
                subchunk_header = chunk_data[offset : offset + 8]
                if len(subchunk_header) < 8:
                     self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Warning: Incomplete subchunk header at offset {offset}.")
                     break
                    
                subchunk_id = subchunk_header[0:4]
                subchunk_size = int.from_bytes(subchunk_header[4:8], byteorder='little')
                data_start = offset + 8
                data_end = data_start + subchunk_size
                
                if data_end > end_position:
                     self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Warning: Subchunk {subchunk_id!r} size ({subchunk_size}) exceeds adtl chunk boundary.")
                     break

                self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Found subchunk: {subchunk_id!r}, size: {subchunk_size}")
                if subchunk_id in (b'labl', b'note'):
                    if subchunk_size < 4:
                         self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Warning: Label/note subchunk too small ({subchunk_size} bytes).")
                    else:
                        cue_id = int.from_bytes(chunk_data[data_start : data_start + 4], byteorder='little')
                        label_data = chunk_data[data_start + 4 : data_end]
                        # Decode label, strip null terminator if present
                        label = label_data.split(b'\0', 1)[0].decode('utf-8', errors='replace')
                        labels[cue_id] = label
                        self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Label for cue {cue_id}: {label}")
                else:
                    self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Skipping unknown adtl subchunk: {subchunk_id!r}")
                
                # Move to the next subchunk, considering padding
                offset = data_end
                if subchunk_size % 2 != 0:
                    offset += 1 # Skip padding byte
                    
        except Exception as e:
            self.log(f"[LogicMarkerReader._process_adtl_chunk_data] Error processing adtl chunk data: {e}")
            
        return labels
        
    def _process_bext_chunk_data(self, chunk_data: bytes) -> Dict[str, Union[str, int]]:
        """Process BWF extension (bext) chunk data."""
        bwf_info = {}
        try:
            # Basic fields from BWF spec EBU Tech 3285
            if len(chunk_data) >= 602: # Minimum size for standard fields
                bwf_info['bwf_description'] = chunk_data[0:256].split(b'\0', 1)[0].decode('utf-8', errors='replace').strip()
                bwf_info['bwf_originator'] = chunk_data[256:288].split(b'\0', 1)[0].decode('utf-8', errors='replace').strip()
                bwf_info['bwf_originator_ref'] = chunk_data[288:320].split(b'\0', 1)[0].decode('utf-8', errors='replace').strip()
                bwf_info['bwf_origination_date'] = chunk_data[320:330].split(b'\0', 1)[0].decode('ascii', errors='replace').strip()
                bwf_info['bwf_origination_time'] = chunk_data[330:338].split(b'\0', 1)[0].decode('ascii', errors='replace').strip()
                # TimeReference is 64-bit unsigned integer (offset in samples since midnight)
                bwf_info['bwf_time_reference_low'] = int.from_bytes(chunk_data[338:342], byteorder='little', signed=False)
                bwf_info['bwf_time_reference_high'] = int.from_bytes(chunk_data[342:346], byteorder='little', signed=False)
                # Combine for full 64-bit value if needed: (high << 32) | low
                bwf_info['bwf_version'] = int.from_bytes(chunk_data[346:348], byteorder='little')
                # Skip UMID (64 bytes), LoudnessValue(2), LoudnessRange(2), MaxTruePeakLevel(2), MaxMomentaryLoudness(2), MaxShortTermLoudness(2)
                bwf_info['bwf_reserved'] = chunk_data[412:602] # 190 bytes reserved
                bwf_info['bwf_coding_history'] = chunk_data[602:].split(b'\0', 1)[0].decode('utf-8', errors='replace').strip()
                
                self.log("[LogicMarkerReader._process_bext_chunk_data] Parsed BWF bext chunk.")
            else:
                self.log(f"[LogicMarkerReader._process_bext_chunk_data] Warning: Bext chunk smaller than expected ({len(chunk_data)} bytes). Skipping parse.")

        except Exception as e:
            self.log(f"[LogicMarkerReader._process_bext_chunk_data] Error processing bext chunk data: {e}")
            
        return {k: v for k, v in bwf_info.items() if v} # Return only non-empty values

    def save_to_csv(self, markers: List[Marker], csv_path: str, use_smpte: bool = False):
        """
        Save markers to a CSV file.
        
        Args:
            markers (List[Marker]): List of markers to save.
            csv_path (str): Path to save the CSV file.
            use_smpte (bool): If True, save positions in SMPTE format instead of samples.
        """
        # Sort markers by position (they might be pre-sorted, but ensure it)
        markers.sort(key=lambda m: m.position)
        
        # Prepare data for CSV
        marker_data = []
        position_field = 'position_smpte' if use_smpte else 'position'
        for marker in markers:
            position_val = marker.position_smpte(self.sample_rate, self.smpte_fps, self.smpte_subframes) if use_smpte else marker.position
            marker_data.append({
                position_field: position_val,
                'label': marker.label
            })
            
        # Write to CSV
        if marker_data:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [position_field, 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(marker_data)
            self.log(f"[LogicMarkerReader.save_to_csv] Saved {len(marker_data)} markers to {csv_path} in {'SMPTE' if use_smpte else 'samples'} format")
        else:
            self.log("[LogicMarkerReader.save_to_csv] No markers to save")

# --- Marker Writer ---
class LogicMarkerWriter(LogicMarkerBase):
    """Class to write marker information to WAV audio files."""
    
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        
    def read_markers_from_csv(self, csv_path: str) -> List[Marker]:
        """
        Read markers from a CSV file.
        Expected format: position,label OR position_smpte,label
        
        Args:
            csv_path (str): Path to the CSV file.
            
        Returns:
            List[Marker]: List of markers read, sorted by original position.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[LogicMarkerWriter.read_markers_from_csv] CSV file not found: {csv_path}")
            
        markers = []
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                fieldnames = reader.fieldnames
                if not fieldnames:
                    self.log("[LogicMarkerWriter.read_markers_from_csv] CSV file is empty or has no header.")
                    return []
                
                is_smpte = 'position_smpte' in fieldnames
                position_field = 'position_smpte' if is_smpte else 'position'
                
                if position_field not in fieldnames:
                     raise ValueError(f"Missing required column: '{position_field}'")
                if 'label' not in fieldnames:
                     self.log("[LogicMarkerWriter.read_markers_from_csv] Warning: 'label' column missing, will use default labels.")


                for i, row in enumerate(reader, 1):
                    try:
                        position_str = row.get(position_field)
                        label = row.get('label', f"Marker {i}") # Default label if missing
                        
                        if position_str is None:
                             self.log(f"[LogicMarkerWriter.read_markers_from_csv] Warning: Missing position value in row {i+1}. Skipping.")
                             continue

                        if is_smpte:
                            # Use the Marker class method for robust parsing
                            marker = Marker.from_smpte(i, position_str, label, self.sample_rate, self.smpte_fps, self.smpte_subframes)
                        else:
                            position = int(position_str)
                            marker = Marker(id=i, position=position, label=label)
                            
                        markers.append(marker)
                        self.log(f"[LogicMarkerWriter.read_markers_from_csv] Read marker {i}: id={marker.id}, position={marker.position}, label='{marker.label}'")
                    except (ValueError, TypeError) as e:
                        self.log(f"[LogicMarkerWriter.read_markers_from_csv] Error reading row {i+1}: {e}. Skipping row.")
                        continue
        except Exception as e:
            self.log(f"[LogicMarkerWriter.read_markers_from_csv] Error reading CSV file: {e}")
            raise # Re-raise after logging

        # Sort by position before returning
        markers.sort(key=lambda m: m.position)
        self.log(f"[LogicMarkerWriter.read_markers_from_csv] Successfully read {len(markers)} markers from {csv_path}")
        return markers
        
    def write_markers_to_file(self, input_audio_path: str, output_audio_path: str, markers: List[Marker]):
        """
        Write markers to a new audio file by rewriting the WAV structure.
        
        Args:
            input_audio_path (str): Path to the input audio file (must be WAV).
            output_audio_path (str): Path to save the output audio file.
            markers (List[Marker]): List of markers to write.
        """
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"[LogicMarkerWriter.write_markers_to_file] Input audio file not found: {input_audio_path}")
        
        file_ext = os.path.splitext(input_audio_path)[1].lower()
        if file_ext != '.wav':
            raise ValueError("[LogicMarkerWriter.write_markers_to_file] Input audio file must be a WAV file for marker writing.")

        # Ensure output path is different or user confirms overwrite? For now, allow same path.
        if input_audio_path == output_audio_path:
            # Create a temporary file to avoid reading and writing to the same file handle simultaneously
            temp_output_path = output_audio_path + ".tmp_write"
            self.log(f"[LogicMarkerWriter.write_markers_to_file] Input and output paths are the same. Using temporary file: {temp_output_path}")
            rewrite_target = temp_output_path
            # Copy original to temp for reading chunks
            shutil.copy2(input_audio_path, rewrite_target)
        else:
            # Copy original to final output path first, then rewrite that copy
             try:
                shutil.copy2(input_audio_path, output_audio_path)
                rewrite_target = output_audio_path
             except Exception as e:
                 self.log(f"[LogicMarkerWriter.write_markers_to_file] Error copying input file to output path: {e}")
                 raise

        # Sort markers by ID for consistent chunk generation (WAV spec doesn't require it, but helps)
        markers.sort(key=lambda m: m.id)

        try:
            self._rewrite_wav_with_markers(rewrite_target, markers)
            
            # If we used a temporary file, rename it to the final output path
            if input_audio_path == output_audio_path:
                os.replace(rewrite_target, output_audio_path) # Atomic rename/replace
                self.log(f"[LogicMarkerWriter.write_markers_to_file] Renamed temporary file to final output: {output_audio_path}")
            
            self.log(f"[LogicMarkerWriter.write_markers_to_file] Successfully wrote {len(markers)} markers to {output_audio_path}")

        except Exception as e:
            self.log(f"[LogicMarkerWriter.write_markers_to_file] Error rewriting WAV file: {e}")
            # Clean up potentially corrupted output/temp file
            if os.path.exists(rewrite_target):
                try:
                    os.remove(rewrite_target)
                    self.log(f"[LogicMarkerWriter.write_markers_to_file] Removed intermediate file: {rewrite_target}")
                except OSError as remove_err:
                     self.log(f"[LogicMarkerWriter.write_markers_to_file] Error removing intermediate file {rewrite_target}: {remove_err}")
            # If output was different from input, and the copy existed, remove it too
            if input_audio_path != output_audio_path and os.path.exists(output_audio_path) and rewrite_target == output_audio_path:
                 try:
                    os.remove(output_audio_path)
                    self.log(f"[LogicMarkerWriter.write_markers_to_file] Removed potentially corrupted output file: {output_audio_path}")
                 except OSError as remove_err:
                     self.log(f"[LogicMarkerWriter.write_markers_to_file] Error removing output file {output_audio_path}: {remove_err}")

            raise # Re-raise the original exception
                
    def _rewrite_wav_with_markers(self, wav_path: str, markers: List[Marker]):
        """
        Rewrites the WAV file at wav_path, replacing existing cue/label chunks.
        
        Args:
            wav_path (str): Path to the WAV file (will be overwritten).
            markers (List[Marker]): List of markers to write.
        """
        self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Starting rewrite for {wav_path}")
        
        # --- 1. Read all existing chunks --- 
        existing_chunks: List[Tuple[bytes, bytes]] = []
        fmt_chunk_found = False
        data_chunk_found = False # Important for valid WAV
        try:
            for chunk_id, chunk_data in self._read_wav_chunks(wav_path):
                existing_chunks.append((chunk_id, chunk_data))
                if chunk_id == b'fmt ':
                    fmt_chunk_found = True
                elif chunk_id == b'data':
                    data_chunk_found = True
        except Exception as e:
            # Error during reading is fatal for rewrite
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Fatal error reading existing chunks: {e}")
            raise
            
        if not fmt_chunk_found or not data_chunk_found:
            raise ValueError(f"Cannot rewrite WAV file: Missing essential chunk(s) - fmt found: {fmt_chunk_found}, data found: {data_chunk_found}")

        # --- 2. Filter out existing marker chunks --- 
        filtered_chunks = []
        for chunk_id, chunk_data in existing_chunks:
            if chunk_id == b'cue ':
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Filtering out existing cue chunk.")
                continue
            # Check for LIST chunk containing adtl
            if chunk_id == b'LIST' and len(chunk_data) >= 4 and chunk_data.startswith(b'adtl'):
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Filtering out existing LIST/adtl chunk.")
                continue
            filtered_chunks.append((chunk_id, chunk_data))
            
        # --- 3. Create new marker chunks --- 
        new_marker_chunks = []
        if markers: # Only create chunks if there are markers to write
            new_cue_chunk_data = self._create_cue_chunk(markers)
            new_label_chunk_data = self._create_label_chunk(markers) # Assumes labels exist or defaults are used
            
            if new_cue_chunk_data:
                new_marker_chunks.append((b'cue ', new_cue_chunk_data))
                self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Created new cue chunk, size {len(new_cue_chunk_data)}.")
            if new_label_chunk_data:
                # _create_label_chunk returns the full LIST chunk data including 'adtl' prefix and size
                list_chunk_data = b'adtl' + new_label_chunk_data
                new_marker_chunks.append((b'LIST', list_chunk_data))
                self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Created new LIST/adtl chunk, total LIST size {len(list_chunk_data)}.")
        else:
             self.log("[LogicMarkerWriter._rewrite_wav_with_markers] No markers provided, writing file without cue/label chunks.")


        # --- 4. Assemble final chunk order ---
        # Standard order: 'fmt ', 'fact' (optional), 'cue ' (optional), 'LIST adtl' (optional), 'data'
        # We aim to insert new marker chunks before 'data'. If 'cue' or 'LIST adtl' exists, they are filtered out.
        
        final_chunks = []
        inserted = False
        # Add chunks before 'data'
        for chunk_id, chunk_data in filtered_chunks:
             if chunk_id == b'data' and not inserted:
                 final_chunks.extend(new_marker_chunks)
                 inserted = True
                 self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Inserted new marker chunks before 'data' chunk.")
             final_chunks.append((chunk_id, chunk_data))

        # If 'data' chunk wasn't found (should have failed earlier), or loop finished without inserting
        if not inserted:
             # This case indicates a structural issue, but as a fallback, append markers
             self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Warning: 'data' chunk not found in expected position or missing. Appending marker chunks.")
             final_chunks.extend(new_marker_chunks)


        # --- 5. Write the new file structure --- 
        try:
            with open(wav_path, 'wb') as f_write:
                f_write.write(b'RIFF')
                f_write.write(b'\0\0\0\0') # Placeholder for RIFF chunk size
                f_write.write(b'WAVE')
                
                self.log("[LogicMarkerWriter._rewrite_wav_with_markers] Writing modified chunks...")
                current_offset = 12 # Start after RIFF header + WAVE ID
                
                for chunk_id, chunk_data in final_chunks:
                    chunk_size = len(chunk_data)
                    f_write.write(chunk_id)
                    f_write.write(struct.pack('<I', chunk_size)) # Chunk size
                    f_write.write(chunk_data)
                    self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Wrote chunk: {chunk_id!r}, size: {chunk_size}")
                    current_offset += 8 + chunk_size # Header + data size
                    
                    # Add padding byte if chunk data size is odd
                    if chunk_size % 2 != 0:
                        f_write.write(b'\0')
                        current_offset += 1
                        self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Added padding byte after {chunk_id!r}")
                
                # Calculate final RIFF chunk size (Total file size - 8 bytes for 'RIFF' ID and size field)
                final_size = f_write.tell()
                riff_chunk_size = final_size - 8
                
                # Go back and write the correct RIFF chunk size
                f_write.seek(4)
                f_write.write(struct.pack('<I', riff_chunk_size))
                self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Successfully rewrote {wav_path}. Final RIFF size: {riff_chunk_size}, Total file size: {final_size}")

        except Exception as e:
            self.log(f"[LogicMarkerWriter._rewrite_wav_with_markers] Error writing modified chunks: {e}")
            # File might be corrupted, rely on the caller's cleanup logic
            raise

    def _create_cue_chunk(self, markers: List[Marker]) -> bytes:
        """Create binary data for the 'cue ' chunk."""
        # Sort by ID for the cue chunk structure (recommended practice)
        markers.sort(key=lambda m: m.id)
        
        num_cues = len(markers)
        chunk_data = struct.pack('<I', num_cues)  # Number of cues
        
        for marker in markers:
            chunk_data += struct.pack('<I', marker.id)        # Cue Point ID
            chunk_data += struct.pack('<I', marker.position)  # Position (play order) - Use sample offset here? Yes, WAV spec uses Sample Offset.
            chunk_data += struct.pack('<4s', b'data')         # Data chunk ID (associated with the 'data' chunk)
            chunk_data += struct.pack('<I', 0)                 # Chunk Start (0 for single 'data' chunk)
            chunk_data += struct.pack('<I', 0)                 # Block Start (0 for uncompressed PCM)
            chunk_data += struct.pack('<I', marker.position)  # Sample Offset (relative to start of data chunk)
        
        return chunk_data
        
    def _create_label_chunk(self, markers: List[Marker]) -> bytes:
        """Create binary data for the 'labl' subchunks within a LIST/adtl chunk."""
        # Returns only the content *inside* the LIST adtl chunk (i.e., the labl subchunks)
        # The 'adtl' prefix and overall LIST size are handled by the caller (_rewrite_wav_with_markers)
        
        # Sort by ID (consistent with cue chunk)
        markers.sort(key=lambda m: m.id)
        
        adtl_content = b''
        for marker in markers:
            label_text = marker.label
            label_bytes = label_text.encode('utf-8') + b'\0' # Null-terminated UTF-8 string
            
            # Size of the 'labl' subchunk content = 4 bytes for cue ID + label bytes
            labl_content_size = 4 + len(label_bytes)
            
            # Build the 'labl' subchunk
            adtl_content += struct.pack('<4s', b'labl')          # Subchunk ID 'labl'
            adtl_content += struct.pack('<I', labl_content_size) # Size of content (cue_id + label_bytes)
            adtl_content += struct.pack('<I', marker.id)         # Associated Cue Point ID
            adtl_content += label_bytes                          # Label text with null terminator
            
            # Pad 'labl' subchunk data to even length if needed (including the header)
            if (labl_content_size + 8) % 2 != 0: # +8 for 'labl' ID and size field
                 adtl_content += b'\0' # This padding belongs *after* the label bytes for this subchunk

        return adtl_content

    def _create_silent_wav(self, output_path: str, duration_samples: int, 
                             channels: int = 1, bit_depth: int = 16):
        """
        Creates a silent WAV file with the specified parameters.

        Args:
            output_path (str): Path to save the WAV file.
            duration_samples (int): Duration of the silence in samples.
            channels (int): Number of audio channels (default: 1 for mono).
            bit_depth (int): Bit depth (default: 16).
        """
        if bit_depth not in [16, 24]: # Only handle common PCM bit depths for simplicity
            raise ValueError("Unsupported bit depth. Only 16 or 24 bits are supported.")
            
        self.log(f"[LogicMarkerWriter._create_silent_wav] Creating silent WAV: {output_path}, duration: {duration_samples} samples, "
                 f"rate: {self.sample_rate}, channels: {channels}, depth: {bit_depth}")

        num_samples = duration_samples
        bytes_per_sample = bit_depth // 8
        byte_rate = self.sample_rate * channels * bytes_per_sample
        block_align = channels * bytes_per_sample
        fmt_chunk_size = 16 # Standard PCM fmt chunk size
        data_size = num_samples * block_align
        
        # RIFF chunk size = 'WAVE' + size(fmt chunk) + size(data chunk)
        # size(chunk) = 8 bytes header + chunk data size
        riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_size)

        try:
            with open(output_path, 'wb') as f:
                # RIFF Header
                f.write(b'RIFF')
                f.write(struct.pack('<I', riff_chunk_size))
                f.write(b'WAVE')
                
                # Format Chunk ('fmt ')
                f.write(b'fmt ')
                f.write(struct.pack('<I', fmt_chunk_size)) # Chunk size
                f.write(struct.pack('<H', 1))              # Audio format (1 = PCM)
                f.write(struct.pack('<H', channels))       # Number of channels
                f.write(struct.pack('<I', self.sample_rate)) # Sample rate
                f.write(struct.pack('<I', byte_rate))       # Byte rate
                f.write(struct.pack('<H', block_align))    # Block align
                f.write(struct.pack('<H', bit_depth))      # Bits per sample
                
                # Data Chunk ('data')
                f.write(b'data')
                f.write(struct.pack('<I', data_size))      # Data size
                
                # Write silence (zeros)
                # Write in chunks to avoid large memory allocation for long durations
                chunk_size = 1024 * 1024 # 1MB chunks
                bytes_written = 0
                while bytes_written < data_size:
                    bytes_to_write = min(chunk_size, data_size - bytes_written)
                    f.write(b'\x00' * bytes_to_write)
                    bytes_written += bytes_to_write
                    
            self.log(f"[LogicMarkerWriter._create_silent_wav] Successfully created silent WAV: {output_path}")
        except Exception as e:
            self.log(f"[LogicMarkerWriter._create_silent_wav] Error creating silent WAV file: {e}")
            # Clean up potentially incomplete file
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass # Ignore cleanup error
            raise # Re-raise the original error

# --- Marker Tester ---
class LogicMarkerTester(LogicMarkerBase):
    """Class for testing marker reading and writing."""
    
    def __init__(self, verbose=True):
        # Ensure Base constructor is called
        super().__init__(verbose=verbose)
        self.reader = LogicMarkerReader(verbose=verbose)
        self.writer = LogicMarkerWriter(verbose=verbose)

    def test_write(self, source_audio: str, target_audio: str):
        """
        Tests marker writing by reading from source, writing to target, reading back, and comparing.
        
        Args:
            source_audio: Path to WAV file with existing markers (ground truth).
            target_audio: Path to WAV file (can be without markers) to write to.
        """
        temp_csv = None
        target_marked_path = None # Define here for cleanup scope

        try:
            print("\n=== [LogicMarkerTester.test_write] Starting Marker Write Test ===")
            
            # Step 1: Read markers from source file
            print(f"\n1. Reading markers from source: {source_audio}")
            source_markers = self.reader.read_markers_from_file(source_audio)
            
            if not source_markers:
                print("[LogicMarkerTester.test_write] Error: No markers found in source file! Cannot perform test.")
                return False # Indicate failure
                
            print(f"[LogicMarkerTester.test_write] Found {len(source_markers)} markers in source file.")
            # Sort by ID for consistent comparison later
            source_markers.sort(key=lambda m: m.id)
            
            # Step 2: (Optional but good practice) Save markers to temporary CSV
            temp_csv = "temp_test_markers.csv"
            print(f"\n2. Saving source markers to temporary CSV: {temp_csv}")
            self.reader.save_to_csv(source_markers, temp_csv) 
            # We will read back from the source_markers list directly for the write step
            # This step just verifies saving works as part of the test flow.

            # Step 3: Write markers to target file
            target_marked_path = target_audio + ".marked_test.wav" # Use a distinct name for the test output
            print(f"\n3. Writing markers to test output file: {target_marked_path}")
            # Use the markers read directly from the source file
            self.writer.write_markers_to_file(target_audio, target_marked_path, source_markers) 
            print(f"[LogicMarkerTester.test_write] Wrote {len(source_markers)} markers to {target_marked_path}.")

            # Step 4: Read back markers from the newly written target file
            print(f"\n4. Reading back markers from: {target_marked_path}")
            result_markers = self.reader.read_markers_from_file(target_marked_path)
             # Sort by ID for consistent comparison
            result_markers.sort(key=lambda m: m.id)
            
            if not result_markers:
                print("[LogicMarkerTester.test_write] Error: No markers found in target file after writing!")
                return False # Indicate failure
            print(f"[LogicMarkerTester.test_write] Found {len(result_markers)} markers in result file.")

            # Step 5: Compare source and result markers
            print("\n5. Comparing source and result markers...")
            
            match = self.compare_marker_lists(source_markers, result_markers)

            print("\n=== Test Results ===")
            if match:
                print("✓ SUCCESS: Markers in the rewritten file match the source markers.")
                return True
            else:
                print("✗ FAILURE: Markers in the rewritten file DO NOT match the source markers.")
                return False

        except FileNotFoundError as e:
             print(f"[LogicMarkerTester.test_write] Test Error: File not found - {e}")
             return False
        except ValueError as e:
             print(f"[LogicMarkerTester.test_write] Test Error: Invalid data or file format - {e}")
             return False
        except Exception as e:
            print(f"[LogicMarkerTester.test_write] Unexpected Test Error: {type(e).__name__} - {e}")
            # Consider adding traceback logging here if needed
            # import traceback
            # traceback.print_exc()
            return False
        finally:
            # Store test success status to decide on cleanup
            test_succeeded = False # Default to False
            try:
                # Check if 'match' variable exists and is True (only set if comparison happened)
                if 'match' in locals() and match:
                    test_succeeded = True
            except NameError: # Should not happen if try block completed, but for safety
                 pass 

            # Cleanup temporary CSV file always
            if temp_csv and os.path.exists(temp_csv):
                try:
                    os.remove(temp_csv)
                    self.log(f"[LogicMarkerTester.test_write] Cleaned up temporary CSV: {temp_csv}")
                except OSError as e:
                    self.log(f"[LogicMarkerTester.test_write] Warning: Could not remove temp CSV {temp_csv}: {e}")
            
            # Cleanup target marked file ONLY if test did NOT succeed
            if not test_succeeded and target_marked_path and os.path.exists(target_marked_path):
                 self.log(f"[LogicMarkerTester.test_write] Test failed or incomplete, cleaning up output file: {target_marked_path}")
                 try:
                    os.remove(target_marked_path)
                 except OSError as e:
                    self.log(f"[LogicMarkerTester.test_write] Warning: Could not remove test output file {target_marked_path}: {e}")
            elif test_succeeded:
                 self.log(f"[LogicMarkerTester.test_write] Test succeeded, keeping output file for validation: {target_marked_path}")

    def compare_marker_lists(self, list1: List[Marker], list2: List[Marker]) -> bool:
        """Compares two lists of Marker objects."""
        
        if len(list1) != len(list2):
            print(f"Mismatch: Number of markers differ (Source: {len(list1)}, Result: {len(list2)})")
            return False

        match = True
        # Assuming lists are sorted by ID
        for m1, m2 in zip(list1, list2):
            if m1.id != m2.id:
                 print(f"Mismatch: Marker ID sequence differs (found {m1.id} vs {m2.id})")
                 # This shouldn't happen if both lists were sorted by ID, indicates bigger issue.
                 match = False
                 # Continue checking other markers if possible
                 continue # Or break? Let's continue to report all differences.

            marker_match = True
            if m1.position != m2.position:
                 print(f"Mismatch: Position for Marker ID {m1.id} (Source: {m1.position}, Result: {m2.position})")
                 marker_match = False
            if m1.label != m2.label:
                 print(f"Mismatch: Label for Marker ID {m1.id} (Source: '{m1.label}', Result: '{m2.label}')")
                 marker_match = False
                 
            if not marker_match:
                match = False # Mark overall comparison as failed

        if match:
             print("Markers match perfectly.")
        
        return match

    # --- Placeholder Tests ---

    def test_write_empty_markers(self, target_audio: str):
        """
        Tests writing an empty list of markers to a file.
        Should verify the output file has no 'cue ' or 'LIST adtl' chunks afterwards.
        Effectively tests marker removal.
        
        Args:
            target_audio: Path to WAV file to write to (will be modified/copied).
        """
        self.log("[LogicMarkerTester.test_write_empty_markers] Test not implemented yet.")
        raise NotImplementedError("Testing writing empty markers is not yet implemented.")

    def test_write_with_special_characters(self, source_audio: str, target_audio: str):
        """
        Tests writing markers with labels containing various special characters.
        Includes Unicode, quotes, commas, etc., to ensure proper encoding, 
        decoding, and CSV handling (if applicable during test).
        
        Args:
            source_audio: Base WAV file (content doesn't matter as much as structure).
            target_audio: Path to write the test file to.
        """
        self.log("[LogicMarkerTester.test_write_with_special_characters] Test not implemented yet.")
        raise NotImplementedError("Testing special characters in labels is not yet implemented.")

    def test_write_duplicate_positions(self, source_audio: str, target_audio: str):
        """
        Tests how writing markers with identical sample positions is handled.
        Checks if the library allows it, and if reading back preserves them correctly.
        
        Args:
            source_audio: Base WAV file.
            target_audio: Path to write the test file to.
        """
        self.log("[LogicMarkerTester.test_write_duplicate_positions] Test not implemented yet.")
        raise NotImplementedError("Testing duplicate marker positions is not yet implemented.")

    def test_read_malformed_wav(self, malformed_audio: str):
        """
        Tests reading from WAV files with intentionally corrupted marker chunks.
        Verifies that the reader handles errors gracefully (e.g., returns empty list, 
        logs warnings) rather than crashing.
        
        Args:
            malformed_audio: Path to a WAV file known to have bad marker chunks.
        """
        self.log("[LogicMarkerTester.test_read_malformed_wav] Test not implemented yet.")
        raise NotImplementedError("Testing reading malformed WAV files is not yet implemented.")

    def test_smpte_conversion_accuracy(self, sample_rate: int, fps: int, subframes: int):
        """
        Tests the accuracy of SMPTE to sample conversion and back.
        Creates markers using Marker.from_smpte, writes/reads (or simulates), 
        converts back using marker.position_smpte, and compares.
        
        Args:
            sample_rate: Sample rate to test with.
            fps: Frames per second for SMPTE.
            subframes: Subframes per frame for SMPTE.
        """
        self.log("[LogicMarkerTester.test_smpte_conversion_accuracy] Test not implemented yet.")
        raise NotImplementedError("Testing SMPTE conversion accuracy is not yet implemented.")
        
    def test_read_csv_variants(self, csv_path: str):
        """
        Tests reading markers from CSV files with various potential issues.
        Checks handling of different delimiters (if supported), header case variations, 
        extra columns, missing labels, etc.
        
        Args:
            csv_path: Path to a CSV file designed for robustness testing.
        """
        self.log("[LogicMarkerTester.test_read_csv_variants] Test not implemented yet.")
        raise NotImplementedError("Testing reading CSV variants is not yet implemented.")

    def test_write_inplace(self, audio_file: str):
        """
        Specifically tests the scenario where input and output paths are the same.
        Reads markers, modifies them slightly (or uses the same), writes back to the 
        *same* file path, reads again, and verifies correctness. Ensures the 
        temporary file mechanism works.

        Args:
            audio_file: Path to a WAV file (will be modified in place).
        """
        self.log("[LogicMarkerTester.test_write_inplace] Test not implemented yet.")
        raise NotImplementedError("Testing in-place writing is not yet implemented.")

# --- Command Line Interface ---
class LogicMarkerCLI:
    """Command line interface using python-fire."""
    
    def read(self, file_path: str, csv_output: Optional[str] = None, smpte: bool = False, 
             verbose: bool = False):
        """
        Extract marker information from a WAV file.
        
        Args:
            file_path: Path to the WAV audio file.
            csv_output: Optional path to save markers as CSV.
            smpte: If True, display/save positions in SMPTE format (default: samples).
            verbose: Enable detailed logging output.
        """
        reader = LogicMarkerReader(verbose=verbose)
        try:
            markers = reader.read_markers_from_file(file_path)
            
            if markers:
                print(f"\n[LogicMarkerCLI.read] Found {len(markers)} markers:")
                # Display markers (sorted by position by the reader)
                for marker in markers:
                    position_str = marker.position_smpte() if smpte else str(marker.position)
                    format_str = "SMPTE" if smpte else "samples"
                    print(f"  ID: {marker.id}, Position ({format_str}): {position_str}, Label: '{marker.label}'")
                    
                if csv_output:
                    reader.save_to_csv(markers, csv_output, use_smpte=smpte)
                    print(f"\n[LogicMarkerCLI.read] Markers saved to CSV: {csv_output}")
            else:
                print("\n[LogicMarkerCLI.read] No markers found in the file.")
                
        except FileNotFoundError:
            print(f"[LogicMarkerCLI.read] Error: Input file not found at '{file_path}'")
            sys.exit(1)
        except ValueError as e:
             print(f"[LogicMarkerCLI.read] Error: Invalid file format or data - {e}")
             sys.exit(1)
        except Exception as e:
            print(f"[LogicMarkerCLI.read] An unexpected error occurred: {type(e).__name__} - {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
            
    def write(self, input_audio: str, csv_file: str, output_audio: str, 
              verbose: bool = False):
        """
        Write markers from a CSV file to a WAV audio file.
        Overwrites existing markers in the output file.
        
        Args:
            input_audio: Path to the source WAV file (used as template).
            csv_file: Path to the CSV file containing markers (position/label or position_smpte/label).
            output_audio: Path to save the output WAV file with markers.
            verbose: Enable detailed logging output.
        """
        writer = LogicMarkerWriter(verbose=verbose)
        try:
            # 1. Read markers from CSV
            print(f"[LogicMarkerCLI.write] Reading markers from CSV: {csv_file}")
            markers_to_write = writer.read_markers_from_csv(csv_file)
            if not markers_to_write:
                 print("[LogicMarkerCLI.write] No markers found in CSV file. Output file will be a copy of the input without markers.")
                 # Allow writing empty markers if user intends to clear them
                 # shutil.copy2(input_audio, output_audio) # The writer handles copying now

            # 2. Write markers to the output audio file
            print(f"[LogicMarkerCLI.write] Writing {len(markers_to_write)} markers from {input_audio} to {output_audio}")
            writer.write_markers_to_file(input_audio, output_audio, markers_to_write)
            
            print(f"\n[LogicMarkerCLI.write] Successfully wrote {len(markers_to_write)} markers to {output_audio}")

        except FileNotFoundError as e:
            print(f"[LogicMarkerCLI.write] Error: File not found - {e}")
            sys.exit(1)
        except (ValueError, TypeError, csv.Error) as e:
             print(f"[LogicMarkerCLI.write] Error: Invalid file format or data - {e}")
             sys.exit(1)
        except Exception as e:
            print(f"[LogicMarkerCLI.write] An unexpected error occurred: {type(e).__name__} - {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def test_write(self, source_audio: str, target_audio: str, verbose: bool = False):
        """
        Test marker writing: Read(source) -> Write(target) -> Read(target) -> Compare.
        
        Args:
            source_audio: Path to WAV file with reference markers.
            target_audio: Path to WAV file (can be without markers) to use as base for writing.
            verbose: Enable detailed logging output.
        """
        tester = LogicMarkerTester(verbose=verbose)
        success = tester.test_write(source_audio, target_audio)
        # Exit with status 0 on success, 1 on failure
        sys.exit(0 if success else 1)

# --- Top-level CLI Functions ---

def create_from_csv(csv_file: str, output_audio: str,
                    channels: int = 1, bit_depth: int = 16, buffer_seconds: float = 1.0,
                    verbose: bool = False):
    """
    Create a silent WAV file and write markers from a CSV file into it.
    Calculates duration based on the last marker + buffer.

    Args:
        csv_file: Path to the CSV file containing markers (position/label or position_smpte/label).
        output_audio: Path to save the output WAV file with silence and markers.
        channels: Number of audio channels for the silent WAV (default: 1).
        bit_depth: Bit depth for the silent WAV (default: 16).
        buffer_seconds: Extra silence duration (in seconds) to add after the last marker (default: 1.0).
        verbose: Enable detailed logging output.
    """
    writer = LogicMarkerWriter(verbose=verbose)
    try:
        # 1. Read markers from CSV
        print(f"[create_from_csv] Reading markers from CSV: {csv_file}") # Updated log prefix
        markers = writer.read_markers_from_csv(csv_file)
        if not markers:
            print(f"[create_from_csv] Error: No markers found in CSV file '{csv_file}'. Cannot create file.") # Updated log prefix
            sys.exit(1)

        # 2. Calculate required duration
        max_position = 0
        if markers: # Should always be true due to check above, but belts and suspenders
            max_position = max(m.position for m in markers)

        buffer_samples = int(buffer_seconds * writer.sample_rate)
        duration_samples = max_position + buffer_samples
        duration_seconds = duration_samples / writer.sample_rate
        print(f"[create_from_csv] Last marker at sample {max_position}.") # Updated log prefix
        print(f"[create_from_csv] Calculated required duration: {duration_samples} samples ({duration_seconds:.2f}s including {buffer_seconds}s buffer).") # Updated log prefix

        # 3. Create silent WAV file
        print(f"[create_from_csv] Creating silent WAV file: {output_audio}") # Updated log prefix
        writer._create_silent_wav(output_audio, duration_samples, channels, bit_depth)

        # 4. Write markers to the silent file (in-place rewrite)
        print(f"[create_from_csv] Writing {len(markers)} markers to {output_audio}") # Updated log prefix
        # Use the output path as both input and output for the writer function
        writer.write_markers_to_file(input_audio_path=output_audio, output_audio_path=output_audio, markers=markers)

        print(f"\n[create_from_csv] Successfully created {output_audio} with {len(markers)} markers.") # Updated log prefix

    except FileNotFoundError as e:
        print(f"[create_from_csv] Error: File not found - {e}") # Updated log prefix
        sys.exit(1)
    except (ValueError, TypeError, csv.Error) as e:
         print(f"[create_from_csv] Error: Invalid file format or data - {e}") # Updated log prefix
         # Clean up potentially created but incomplete output file
         if os.path.exists(output_audio):
             try: os.remove(output_audio)
             except OSError: pass
         sys.exit(1)
    except Exception as e:
        print(f"[create_from_csv] An unexpected error occurred: {type(e).__name__} - {e}") # Updated log prefix
        # Clean up potentially created but incomplete output file
        if os.path.exists(output_audio):
             try: os.remove(output_audio)
             except OSError: pass
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def convert_marker_format(input_csv: str, output_csv: str, target_format: str,
                          sample_rate: int = DEFAULT_SAMPLE_RATE, fps: int = DEFAULT_SMPTE_FPS, subframes: int = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME,
                          verbose: bool = False):
    """
    Convert a marker CSV file between sample position and SMPTE timecode formats.

    Args:
        input_csv: Path to the input CSV file containing markers.
        output_csv: Path to save the converted CSV file.
        target_format: The desired output format ('smpte' or 'samples').
        sample_rate: Sample rate to use for conversion (default: 48000).
        fps: Frames per second for SMPTE conversion (default: 30).
        subframes: Subframes per frame for SMPTE conversion (default: 100).
        verbose: Enable detailed logging output.
    """
    if verbose: print("[convert_marker_format] Function not implemented yet.")
    raise NotImplementedError("Converting marker formats is not yet implemented.")

def offset_markers(input_csv: str, output_csv: str, offset: str, offset_unit: str,
                   sample_rate: int = DEFAULT_SAMPLE_RATE, fps: int = DEFAULT_SMPTE_FPS, subframes: int = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME,
                   verbose: bool = False):
    """
    Shift the time position of all markers in a CSV file by a specified offset.

    Args:
        input_csv: Path to the input CSV file containing markers.
        output_csv: Path to save the CSV file with offset markers.
        offset: The offset amount (e.g., "+1000", "-0:0:1:15.0", "5.5"). Needs sign.
        offset_unit: The unit of the offset ('samples', 'smpte', or 'seconds').
        sample_rate: Sample rate if converting offset or positions (default: 48000).
        fps: Frames per second if using SMPTE (default: 30).
        subframes: Subframes per frame if using SMPTE (default: 100).
        verbose: Enable detailed logging output.
    """
    if verbose: print("[offset_markers] Function not implemented yet.")
    raise NotImplementedError("Offsetting markers is not yet implemented.")

def remove_markers(input_audio: str, output_audio: str, verbose: bool = False):
    """
    Create a copy of a WAV file with all marker information removed.

    Args:
        input_audio: Path to the source WAV file.
        output_audio: Path to save the WAV file without markers.
        verbose: Enable detailed logging output.
    """
    writer = LogicMarkerWriter(verbose=verbose) # Need writer for file copy/rewrite logic
    try:
        # We can leverage the existing write mechanism with an empty list
        print(f"[remove_markers] Removing markers from {input_audio} to create {output_audio}")
        writer.write_markers_to_file(input_audio, output_audio, [])
        print(f"[remove_markers] Successfully created {output_audio} without markers.")
    except FileNotFoundError as e:
        print(f"[remove_markers] Error: File not found - {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
         print(f"[remove_markers] Error: Invalid file format or data - {e}")
         sys.exit(1)
    except Exception as e:
        print(f"[remove_markers] An unexpected error occurred: {type(e).__name__} - {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def validate_markers(file_path: str, verbose: bool = False):
    """
    Validate marker information in a WAV file or format of a CSV file.

    Args:
        file_path: Path to the WAV or CSV file to validate.
        verbose: Enable detailed logging output.
    """
    if verbose: print("[validate_markers] Function not implemented yet.")
    raise NotImplementedError("Validating marker files is not yet implemented.")

def stretch_markers(input_csv: str, output_csv: str, factor: float, anchor_time: str = "0", anchor_unit: str = 'samples',
                    sample_rate: int = DEFAULT_SAMPLE_RATE, fps: int = DEFAULT_SMPTE_FPS, subframes: int = DEFAULT_SMPTE_SUBFRAMES_PER_FRAME,
                    verbose: bool = False):
    """
    Stretch or compress marker positions relative to an anchor time by a given factor.

    Args:
        input_csv: Path to the input CSV file containing markers.
        output_csv: Path to save the CSV file with stretched markers.
        factor: The stretch factor (e.g., 2.0 for double speed, 0.5 for half speed).
        anchor_time: The time point around which stretching occurs (default: "0").
        anchor_unit: The unit of the anchor time ('samples', 'smpte', or 'seconds', default: 'samples').
        sample_rate: Sample rate for time conversions (default: 48000).
        fps: Frames per second if using SMPTE (default: 30).
        subframes: Subframes per frame if using SMPTE (default: 100).
        verbose: Enable detailed logging output.
    """
    if verbose: print("[stretch_markers] Function not implemented yet.")
    raise NotImplementedError("Stretching markers is not yet implemented.")

# --- Main Execution ---

def main():
    # Set default encoding to UTF-8 for robustness with labels?
    # Not usually necessary for modern Python 3, but consider if issues arise.
    # sys.stdout.reconfigure(encoding='utf-8')
    # Expose the class methods and the top-level functions
    fire.Fire({
        'cli': LogicMarkerCLI,
        'create_from_csv': create_from_csv,
        'convert_marker_format': convert_marker_format,
        'offset_markers': offset_markers,
        'remove_markers': remove_markers,
        'validate_markers': validate_markers,
        'stretch_markers': stretch_markers
    })

if __name__ == '__main__':
    main()
