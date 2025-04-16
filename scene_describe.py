"""
Uses Scenecut Detect to identify scenes in a video, describes the scenes using a 
visual LLM, allows merging of similar scenes, and outputs an enriched CSV file.

Examples:
  python scene_describe.py process \\
    --video_path video.mp4 \\
    --output_dir scene_output \\
    --llm_context context.txt \\
    --verbose

  python scene_describe.py clear_scene_descriptions --scene_data scene_output/scenes_data.json
"""

import csv
import logging
import os
import fire
from typing import List, Dict, Any, Optional, Tuple
import json # Added import

# Scenedetect imports
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector # Example detector
from scenedetect.scene_manager import save_images
from scenedetect.video_stream import VideoOpenFailure
from scenedetect.stats_manager import StatsFileCorrupt

# Google Generative AI and Image handling
import google.generativeai as genai
from PIL import Image

# --- LLM Prompt Constants ---

# Template for generating the main scene description
LLM_DESCRIPTION_BASE_PROMPT = """Analyze these three movie images (start, middle, end of a scene) and provide a description of the scene's setting, action, and any notable visual elements.
Focus only on what is visible in the images.
If additional context is provided, use it to inform the description.
The description should be suitable for usage in cue sheets for sound and music departments.
Use prose only, no markdown, headings, bullets, or other formatting.
Write in a clear and concise style, avoiding unnecessary adjectives and adverbs.
If the scene is just a black screen, describe it as "Black screen".
Maximum description length is 100 words."""

# Suffix to add to the description prompt when context is provided
LLM_DESCRIPTION_CONTEXT_SUFFIX = """
--- Provided Context ---
{user_context}
--- End Context ---"""

# Template for generating title and summary from a description
LLM_TITLE_SUMMARY_PROMPT_TEMPLATE = f"""Analyze the following scene description and generate a concise title (under 30 words) 
and a short summary (under 100 characters, strictly). 
The title and summary should be suitable for usage in cue sheets for sound and music departments.
Do not use colons or semicolons in the title or summary.
There should be no repetition of words across the title and summary.
If the scene is just a black screen, title it "Black screen" and summarize as "Black screen".
Respond ONLY with a JSON object containing two keys: 'title' and 'summary'.
Description:\\n{{description}}""" # Note: Escaped braces for f-string

# Template for checking scene similarity based on descriptions
LLM_SIMILARITY_PROMPT_TEMPLATE = f"""Compare the following two scene descriptions. Based ONLY on these descriptions, 
are the scenes they describe likely to be from the same scene? Answer ONLY with 'yes' or 'no'.
Description 1 (Scene {{scene_num1}}):\\n{{desc1}}
Description 2 (Scene {{scene_num2}}):\\n{{desc2}}""" # Note: Escaped braces for f-string

# --- End LLM Prompt Constants ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: [%(name)s.%(funcName)s] %(message)s')
# Get logger instance for the class
logger = logging.getLogger(f'{__name__}.SceneDescriber') 

class SceneDescriber:
    """Orchestrates scene detection, description, merging, and output generation."""

    def __init__(self, verbose: bool = False):
        """
        Initializes the SceneDescriber.

        Args:
            verbose: If True, sets logging level to DEBUG.
        """
        self._configure_logging(verbose)
        # Update logger name after configuration potentially changes level
        global logger
        logger = logging.getLogger(f'{self.__class__.__name__}') 

        # Configure Google AI API Key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY environment variable not set. LLM features will fail.")
            # Optionally, raise an error or provide a way to pass the key as an argument
            # raise ValueError("GOOGLE_API_KEY must be set in the environment")
        else:
            genai.configure(api_key=api_key)
            logger.info("Google Generative AI configured.")

        logger.debug("SceneDescriber initialized.")

    def _configure_logging(self, verbose: bool):
        """Sets the logging level."""
        # Get the root logger and set its level
        root_logger = logging.getLogger()
        if verbose:
            root_logger.setLevel(logging.DEBUG)
            # Re-fetch the class logger in case levels were changed
            class_logger = logging.getLogger(f'{self.__class__.__name__}') 
            class_logger.debug("Verbose logging enabled.")
        else:
            root_logger.setLevel(logging.INFO)
            
    def _format_scene_list(self, scene_list_tuples: List[Tuple[FrameTimecode, FrameTimecode]], base_timecode: FrameTimecode) -> List[Dict[str, Any]]:
        """Converts scenedetect scene list tuples to the dictionary format."""
        formatted_scenes = []
        for i, (start, end) in enumerate(scene_list_tuples):
             scene_num = i + 1
             start_secs = start.get_seconds()
             end_secs = end.get_seconds()
             length_secs = end_secs - start_secs
             # Ensure end frame is inclusive for length calculation if needed, scenedetect might be exclusive
             # Frame numbers from FrameTimecode are 0-based, adjust if 1-based needed elsewhere
             start_frame = start.get_frames()
             end_frame = end.get_frames() -1 # Adjust for inclusive end frame in calculations
             length_frames = end_frame - start_frame + 1

             # Calculate length timecode (requires base_timecode for framerate)
             # Create a FrameTimecode representing the duration
             duration_frames = length_frames 
             length_timecode_obj = base_timecode + duration_frames
             length_timecode_str = length_timecode_obj.get_timecode()


             scene_data = {
                 'Scene Number': scene_num,
                 'Start Frame': start_frame + 1, # Convert to 1-based index for output consistency
                 'Start Timecode': start.get_timecode(),
                 'Start Time (seconds)': start_secs,
                 'End Frame': end_frame + 1, # Convert to 1-based index
                 'End Timecode': (base_timecode + end_frame).get_timecode(), # Calculate end timecode correctly
                 'End Time (seconds)': end_secs,
                 'Length (frames)': length_frames,
                 'Length (timecode)': length_timecode_str, 
                 'Length (seconds)': length_secs,
                 # Placeholders for descriptions to be added later
                 'Scene Title': '',
                 'Scene Summary': '',
                 'Scene Description': ''
             }
             formatted_scenes.append(scene_data)
        logger.debug(f"Formatted {len(formatted_scenes)} scenes into dictionary list.")
        return formatted_scenes

    def _detect_scenes_and_save_images(self, video_path: str, output_dir: str, detector_threshold: float = 27.0) -> List[Dict[str, Any]]:
        """
        Detects scenes using the scenedetect library, saves images, 
        and returns the scene list in dictionary format.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save the images and scene data JSON.
            detector_threshold: Threshold for the ContentDetector (default: 27.0).

        Returns:
            A list of dictionaries, where each dictionary represents a scene.
        
        Raises:
            VideoOpenFailure: If the video cannot be opened.
            FileNotFoundError: If video_path does not exist.
            Exception: For other scenedetect errors.
        """
        logger.info(f"Detecting scenes and saving images/data for {video_path} to {output_dir}")
        if not os.path.exists(video_path):
             logger.error(f"Video file not found: {video_path}")
             raise FileNotFoundError(f"Video file not found: {video_path}")
             
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open video, create scene manager, add detector
            video = open_video(video_path)
            # Get base timecode for accurate duration calculations later
            base_timecode = video.base_timecode 
            
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=detector_threshold))

            # Perform scene detection
            logger.debug(f"Starting scene detection with threshold {detector_threshold}...")
            scene_manager.detect_scenes(video=video, show_progress=True) # show_progress adds a TQDM bar

            # Get scene list as [(start, end), ...] FrameTimecode tuples
            scene_list_tuples = scene_manager.get_scene_list()
            logger.info(f"Detected {len(scene_list_tuples)} scenes.")

            if not scene_list_tuples:
                 logger.warning("No scenes detected.")
                 return []

            # Save images for detected scenes (start, middle, end)
            logger.info(f"Saving scene images to {output_dir}...")
            save_images(
                scene_list=scene_list_tuples,
                video=video,
                num_images=3,
                output_dir=output_dir, # Use output_dir here
                image_name_template='Scene-$SCENE_NUMBER-$IMAGE_NUMBER' # Matches expected pattern
            )
            logger.debug("Finished saving images.")
            
            # Format the scene list into the desired dictionary structure
            formatted_scenes = self._format_scene_list(scene_list_tuples, base_timecode)
            
            # Save formatted scenes to JSON for potential resume
            json_path = os.path.join(output_dir, 'scenes_data.json') # Save JSON in output_dir
            try:
                with open(json_path, 'w') as f:
                    json.dump(formatted_scenes, f, indent=4)
                logger.info(f"Saved formatted scene data to {json_path}")
            except Exception as e:
                 logger.error(f"Failed to save formatted scene data to {json_path}: {e}")
                 # Decide if this should be a fatal error or just a warning
                 # raise # Or just log the warning and continue

            return formatted_scenes

        except VideoOpenFailure as e:
            logger.error(f"Failed to open video {video_path}: {e}")
            raise
        except StatsFileCorrupt as e:
             logger.error(f"Stats file is corrupt: {e}")
             raise # Or handle differently if stats files are used later
        except Exception as e:
            logger.error(f"An error occurred during scene detection or image saving: {e}")
            raise
        finally:
             # Ensure video object resources are released if open_video succeeded
             if 'video' in locals() and video:
                 try:
                      # Use release() instead of close() for VideoStreamCv2
                      video.release() 
                      logger.debug("Video resources released.")
                 except Exception as e:
                      logger.warning(f"Error releasing video object: {e}")

    def _load_formatted_scenes_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Loads the formatted scene list from a JSON file."""
        logger.info(f"Attempting to load scene data from {json_path}")
        try:
            with open(json_path, 'r') as f:
                loaded_scenes = json.load(f)
            logger.info(f"Successfully loaded {len(loaded_scenes)} scenes from {json_path}")
            return loaded_scenes
        except FileNotFoundError:
            logger.error(f"Scene data file not found: {json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {json_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load scene data from {json_path}: {e}")
            raise

    def _save_scenes_to_json(self, scenes: List[Dict[str, Any]], json_path: str):
        """Saves the scene list (potentially with partial LLM results) to a JSON file."""
        logger.debug(f"Saving {len(scenes)} scenes' data to {json_path}")
        try:
            with open(json_path, 'w') as f:
                json.dump(scenes, f, indent=4)
            logger.info(f"Successfully saved scene data snapshot to {json_path}")
        except Exception as e:
            logger.error(f"Failed to save scene data snapshot to {json_path}: {e}")
            # Decide if this should be a fatal error. Probably should warn and continue.

    def _find_scene_images(self, scene_data: Dict[str, Any], output_dir: str) -> List[str]:
        """Finds the start, middle, and end images for a given scene."""
        scene_number = scene_data['Scene Number']
        # Uses the standard pattern from save_images with image_name_template
        image_patterns = [
            os.path.join(output_dir, f"Scene-{scene_number:03d}-01.jpg"), # Start
            os.path.join(output_dir, f"Scene-{scene_number:03d}-02.jpg"), # Middle
            os.path.join(output_dir, f"Scene-{scene_number:03d}-03.jpg")  # End
        ]
        # Allow for different extensions if save_images format changes (e.g., png, webp)
        # This basic check assumes jpg for now. A more robust check could list dir and regex match.
        found_images = []
        for pattern_base in [f"Scene-{scene_number:03d}-{i:02d}" for i in range(1, 4)]:
             # Check for common image extensions
             for ext in ['.jpg', '.png', '.webp']:
                 img_path = os.path.join(output_dir, pattern_base + ext) # Use output_dir
                 if os.path.exists(img_path):
                     found_images.append(img_path)
                     break # Assume only one extension per image number

        if not found_images:
            logger.warning(f"No images found for Scene {scene_number} in {output_dir} using patterns like {image_patterns[0]}")
        else:
             logger.debug(f"Found images for Scene {scene_number}: {found_images}")
             
        return found_images

    def _get_llm_description(self, image_paths: List[str], user_context: Optional[str] = None) -> str:
        """Gets a detailed description for a scene using the Gemini Vision model."""
        logger.debug(f"Getting LLM description for {len(image_paths)} images with context: '{user_context}'")
        if not image_paths:
            logger.warning("No images provided to LLM for description.")
            return "No images available for description."

        try:
            # Ensure API key is configured (check if genai is configured)
            # A more robust check might be needed if genai doesn't raise an obvious error
            if not os.environ.get("GOOGLE_API_KEY"):
                 error_msg = "Google API Key not configured. Cannot generate description."
                 logger.error(error_msg)
                 # return "Error: LLM not configured."
                 raise ValueError(error_msg) # Raise error

            # Use a current Gemini vision model
            model = genai.GenerativeModel('gemini-1.5-flash-latest') 
            
            # Prepare images for the API
            pil_images = []
            # Define resizing constants
            MAX_PIXELS = 1_150_000 
            MAX_DIMENSION = 1568

            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    original_size = img.size
                    width, height = img.size
                    
                    # --- Image Resizing Logic ---
                    resized = False
                    # 1. Check individual dimensions
                    if width > MAX_DIMENSION or height > MAX_DIMENSION:
                        logger.debug(f"Image {os.path.basename(img_path)} ({width}x{height}) exceeds max dimension ({MAX_DIMENSION}). Resizing.")
                        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.Resampling.LANCZOS)
                        width, height = img.size # Update dimensions after thumbnail
                        resized = True

                    # 2. Check total pixels
                    if width * height > MAX_PIXELS:
                        if not resized: # Log only if it wasn't already logged for dimension resize
                             logger.debug(f"Image {os.path.basename(img_path)} ({width}x{height}) exceeds max pixels ({MAX_PIXELS}). Resizing.")
                        # Calculate scaling factor needed to get under MAX_PIXELS
                        ratio = (MAX_PIXELS / (width * height)) ** 0.5
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        # Use resize for this step as thumbnail might not shrink enough if only slightly over pixel limit
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        resized = True
                        
                    if resized:
                        logger.info(f"Resized image {os.path.basename(img_path)} from {original_size} to {img.size}")
                    # --- End Resizing Logic ---
                        
                    pil_images.append(img)
                except FileNotFoundError:
                    logger.warning(f"Image file not found: {img_path}, skipping.")
            
            if not pil_images:
                 error_msg = "Could not load any images for description."
                 logger.error(error_msg)
                 # return "Error: Could not load images."
                 raise ValueError(error_msg) # Raise error

            # Construct the prompt
            prompt_parts = [
                LLM_DESCRIPTION_BASE_PROMPT
            ]
            if user_context:
                # Ensure context is clearly delineated
                # Format the context suffix with the actual user context
                context_part = LLM_DESCRIPTION_CONTEXT_SUFFIX.format(user_context=user_context)
                prompt_parts.append(context_part)
            
            # Add images to the prompt parts
            prompt_parts.extend(pil_images)

            # Generate content
            logger.debug(f"Sending {len(pil_images)} images and prompt to Gemini Vision...")
            response = model.generate_content(prompt_parts)

            # Extract text - handle potential lack of text or other issues
            if response.parts:
                description = response.text
                logger.info(f"Successfully generated description for scene.") # Avoid logging full desc
                return description
            else:
                 # Log detailed response parts if helpful for debugging
                 logger.error(f"LLM response did not contain expected text parts. Response: {response}")
                 # return "Error: Could not generate description from LLM response."
                 raise ValueError("LLM response did not contain expected text parts.") # Raise error
                 
        except FileNotFoundError as e:
            # This might catch Image.open errors if the path was invalid earlier
            logger.error(f"Image file not found during LLM description generation: {e}")
            # return f"Error: Image file not found - {e}"
            raise # Re-raise the exception
        except Exception as e:
            logger.error(f"An error occurred calling the Google Generative AI API: {e}")
            raise

    def _generate_title_summary(self, description: str) -> (str, str):
        """Generates a title and a short summary from a detailed description using Gemini Pro."""
        logger.debug("Generating title and summary from description.")
        
        # Default values in case of error
        default_title = "Error: Title Generation Failed"
        default_summary = "Error: Summary Generation Failed"

        if description.startswith("Error:"):
             logger.warning(f"Skipping title/summary generation due to previous error: {description}")
             # return default_title, description # Return the error as summary
             raise ValueError(f"Invalid description provided for title/summary generation: {description}") # Raise error
             
        try:
            # Ensure API key is configured
            if not os.environ.get("GOOGLE_API_KEY"):
                 error_msg = "Google API Key not configured. Cannot generate title/summary."
                 logger.error(error_msg)
                 # return default_title, default_summary
                 raise ValueError(error_msg) # Raise error
                 
            # Use a current Gemini text model
            model = genai.GenerativeModel('gemini-1.5-flash-latest') 

            # Format the prompt using the template and the description
            prompt = LLM_TITLE_SUMMARY_PROMPT_TEMPLATE.format(description=description)

            logger.debug("Sending description to Gemini Pro for title/summary...")
            response = model.generate_content(prompt)

            # Attempt to parse the JSON response
            try:
                # Gemini might wrap the JSON in markdown ```json ... ```
                json_text = response.text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:-3].strip() # Remove markdown code block
                elif json_text.startswith("{") and json_text.endswith("}"): # Assume it's just JSON
                     pass # Use json_text as is
                else:
                     raise ValueError(f"Response is not in expected JSON format: {response.text[:100]}...")
                     
                import json # Import json locally for parsing
                result = json.loads(json_text)
                
                title = result.get('title', default_title) 
                summary = result.get('summary', default_summary)

                # Basic validation (optional, but good practice)
                if not isinstance(title, str) or not isinstance(summary, str):
                     logger.warning("Parsed JSON has incorrect types for title/summary.")
                     title = default_title if not isinstance(title, str) else title
                     summary = default_summary if not isinstance(summary, str) else summary
                
                # Enforce summary length strictly
                if len(summary) > 100 and not summary.startswith("Error:"):
                     logger.warning(f"LLM generated summary exceeding 100 chars ({len(summary)}), truncating.")
                     summary = summary[:100]

                logger.info(f"Generated Title: '{title}', Summary: '{summary[:30]}...'")
                return title, summary

            except (json.JSONDecodeError, ValueError, AttributeError, KeyError) as e:
                logger.error(f"Failed to parse LLM response for title/summary: {e}. Response text: {response.text}")
                # Fallback: Maybe try a simpler extraction or just return defaults
                # Simple fallback: Use first line as title, next part as summary (less reliable)
                # lines = response.text.strip().split('\\n')
                # fallback_title = lines[0] if lines else default_title
                # fallback_summary = ' '.join(lines[1:])[:100] if len(lines) > 1 else default_summary
                # logger.warning(f"Using fallback title/summary: '{fallback_title}', '{fallback_summary}'")
                # return fallback_title, fallback_summary
                raise ValueError(f"Failed to parse LLM title/summary response: {e}") # Raise error

        except Exception as e:
            logger.error(f"An error occurred calling the Google Generative AI API for title/summary: {e}")
            # return default_title, default_summary
            raise # Re-raise exception

    def _are_scenes_similar(self, desc1: str, desc2: str, scene_num1: int, scene_num2: int) -> bool:
        """Determines if two scenes are visually similar based on their descriptions using Gemini Pro."""
        logger.debug(f"Comparing similarity between Scene {scene_num1} and Scene {scene_num2}")

        # Avoid LLM call if descriptions are identical or indicate errors
        if desc1 == desc2:
            logger.info(f"Descriptions for Scene {scene_num1} and {scene_num2} are identical. Marking as similar.")
            return True
        if desc1.startswith("Error:") or desc2.startswith("Error:"):
            logger.warning(f"Cannot compare similarity due to error in description(s) for Scene {scene_num1}/{scene_num2}. Marking as not similar.")
            # return False
            raise ValueError(f"Invalid description provided for similarity check: Scene {scene_num1}='{desc1[:50]}...', Scene {scene_num2}='{desc2[:50]}...'") # Raise error
            
        try:
            # Ensure API key is configured
            if not os.environ.get("GOOGLE_API_KEY"):
                 logger.error("Google API Key not configured. Cannot check similarity.")
                 # return False # Assume not similar if LLM is unavailable
                 raise ValueError("Google API Key not configured. Cannot check similarity.") # Raise error

            # Use a current Gemini text model
            model = genai.GenerativeModel('gemini-1.5-flash-latest') 

            # Format the prompt using the template and the provided variables
            prompt = LLM_SIMILARITY_PROMPT_TEMPLATE.format(
                scene_num1=scene_num1,
                desc1=desc1,
                scene_num2=scene_num2,
                desc2=desc2
            )

            logger.debug(f"Sending descriptions for Scene {scene_num1} & {scene_num2} to Gemini Pro for similarity check...")
            response = model.generate_content(prompt)
            
            # Process the response
            answer = response.text.strip().lower()
            is_similar = (answer == 'yes')
            
            logger.info(f"Similarity check result for Scene {scene_num1} & {scene_num2}: '{answer}' -> {'Similar' if is_similar else 'Not Similar'}")
            return is_similar

        except Exception as e:
            logger.error(f"An error occurred calling the Google Generative AI API for similarity check: {e}")
            # Default to not similar in case of error
            # return False
            raise # Re-raise exception

    def _merge_scenes(self, scene1: Dict[str, Any], scene2: Dict[str, Any]) -> Dict[str, Any]:
        """Merges scene2's timing info into scene1."""
        logger.info(f"Merging Scene {scene2['Scene Number']} into Scene {scene1['Scene Number']}")
        
        merged_scene = scene1.copy() # Start with scene1's data
        
        # Update end times and lengths using data directly from scene2 dict
        merged_scene['End Frame'] = scene2['End Frame']
        merged_scene['End Timecode'] = scene2['End Timecode']
        merged_scene['End Time (seconds)'] = scene2['End Time (seconds)']
        
        # Recalculate lengths based on merged start/end
        # Ensure start/end frames are treated consistently (e.g., both 1-based)
        merged_scene['Length (frames)'] = merged_scene['End Frame'] - merged_scene['Start Frame'] + 1 
        merged_scene['Length (seconds)'] = merged_scene['End Time (seconds)'] - merged_scene['Start Time (seconds)']

        # Recalculate length timecode string (requires framerate, which we don't have easily here)
        # Approximate or combine strings as placeholder. A better way requires passing framerate through.
        # TODO: Improve length timecode calculation for merged scenes if framerate becomes available
        merged_scene['Length (timecode)'] = f"{scene1['Length (timecode)']}~{scene2['Length (timecode)']}" # Indicate merged length

        
        # Combine descriptions (or potentially ask LLM to refine)
        # Simple concatenation for now
        merged_scene['Scene Description'] = f"{scene1.get('Scene Description', '')}\\n---\\n{scene2.get('Scene Description', '')}"
        
        # Need to regenerate title/summary for the merged description
        # This will now raise an exception if the merged description is invalid or LLM fails
        # The exception should be caught by the main process loop
        merged_title, merged_summary = self._generate_title_summary(merged_scene['Scene Description'])
        merged_scene['Scene Title'] = merged_title
        merged_scene['Scene Summary'] = merged_summary
        
        logger.debug(f"Merged scene {scene1['Scene Number']} now ends at {merged_scene['End Timecode']}")
        return merged_scene

    def process(self, 
                video_path: str, # Changed: video_path is now required
                output_dir: str, 
                llm_context: Optional[str] = None,
                detector_threshold: float = 27.0, # Added: Allow configuring detector
                llm_resume: bool = True # Added: Allow resuming LLM processing
                ):
        """
        Full processing pipeline: Detect scenes, describe, merge, and write output.
        Outputs are saved to the specified output_dir:
        - Scene images (e.g., Scene-XXX-YY.jpg)
        - Scene data snapshot (scenes_data.json)
        - Final processed data (scenes.csv)

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save scene images, data JSON, and the output scenes.csv.
            llm_context: Optional text context to provide to the visual LLM,
                         or a path to a .txt file containing the context.
            detector_threshold: Threshold for the ContentDetector (default: 27.0).
            llm_resume: If True (default), skips LLM processing for scenes that
                        already have description/title/summary in scenes_data.json.
            # verbose is handled by __init__
        """
        logger.info("Starting full scene processing pipeline.")

        # --- Ensure output directory exists ---
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")

        # Construct full paths for JSON and CSV outputs
        scene_data_json_path = os.path.join(output_dir, 'scenes_data.json')
        output_csv_path = os.path.join(output_dir, 'scenes.csv') # Hardcoded filename
        logger.debug(f"Scene data JSON path: {scene_data_json_path}")
        logger.debug(f"Output CSV path: {output_csv_path}")
        # --- End Path Construction ---


        # Determine actual LLM context (from string or file)
        actual_llm_context = None
        if llm_context:
            # First, check if it looks like a file path
            # if llm_context.lower().endswith(".txt"): # Removed check for specific extension
            # It looks like a file path, try to load it
            try:
                # Check existence explicitly first to give a clear error
                if not os.path.exists(llm_context):
                     logger.error(f"LLM context file not found: {llm_context}")
                     raise FileNotFoundError(f"LLM context file not found: {llm_context}")

                with open(llm_context, 'r') as f:
                    actual_llm_context = f.read().strip()
                logger.info(f"Loaded LLM context from file: {llm_context}")
                # Add check for empty context
                if not actual_llm_context:
                    logger.warning(f"Context file {llm_context} was empty. Proceeding without LLM context.")
                    actual_llm_context = None # Ensure it's treated as no context
            except Exception as e: # Catch other potential reading errors
                logger.error(f"Could not read LLM context file {llm_context}: {e}")
                raise # Re-raise other exceptions
            # else: # Removed this else block entirely
            #     # It doesn't end with .txt, treat as string context
            #     actual_llm_context = llm_context
            #     logger.info("Using provided string as LLM context.")
        else:
             logger.info("No LLM context file path provided.")


        # --- Resume/Detection Logic ---
        # Check if output dir and scene data JSON exist
        should_resume = os.path.isdir(output_dir) and os.path.isfile(scene_data_json_path)
        
        if should_resume:
            logger.info(f"Found existing output directory ({output_dir}) and scene data ({scene_data_json_path}). Attempting to resume.")
            try:
                 raw_scenes = self._load_formatted_scenes_from_json(scene_data_json_path)
                 # Simple check: Does output dir contain files? (More specific checks could be added) 
                 # Check specifically for image files maybe? For now, any file indicates content.
                 if not any(os.scandir(output_dir)):
                      logger.warning(f"Output directory {output_dir} exists but is empty. Scene detection might be needed.")
                      # Decide how to handle this: Proceed with empty images, or force re-detect?
                      # Forcing re-detection for now if dir is empty:
                      should_resume = False 
                      logger.info("Output directory is empty, forcing scene detection.")
                 else:
                      logger.info("Resuming processing with loaded scene data.")
                      
            except Exception as e:
                 logger.error(f"Failed to load scene data for resume: {e}. Running scene detection instead.")
                 should_resume = False
        
        if not should_resume:
            logger.info("Running scene detection and image saving.")
            # Detect scenes and get the list in dictionary format
            try:
                raw_scenes = self._detect_scenes_and_save_images(video_path, output_dir, detector_threshold)
            except (FileNotFoundError, VideoOpenFailure, Exception) as e:
                 logger.error(f"Scene detection failed: {e}. Aborting pipeline.")
                 return # Stop processing if detection fails
        # --- End Resume Logic ---

        if not raw_scenes:
            logger.error("No scenes detected or loaded, cannot proceed.")
            return

        processed_scenes = []
        logger.info("Processing scenes: Describing and checking for merges...")
        
        # Variable to store the first error encountered in the loop
        error_to_raise_after_loop = None 

        try: # Outer try for non-loop errors (like loading video/initial json)
            for i, scene in enumerate(raw_scenes):
                scene_num = scene.get('Scene Number') 
                logger.debug(f"Processing Scene {scene_num}")
                
                scene_processed_successfully = False
                current_error = None # Track error within this iteration

                try:
                    # --- Core Scene Processing Logic ---
                    # Check if we should skip LLM processing
                    has_description = bool(scene.get('Scene Description'))
                    has_title = bool(scene.get('Scene Title'))
                    has_summary = bool(scene.get('Scene Summary'))
                    is_complete = has_description and has_title and has_summary
                    should_skip_llm = llm_resume and is_complete

                    # Get potentially existing data (even if skipping LLM)
                    description = scene.get('Scene Description')
                    title = scene.get('Scene Title')
                    summary = scene.get('Scene Summary')

                    if should_skip_llm:
                        logger.info(f"Resuming Scene {scene_num}: Found existing description, title, and summary. Skipping LLM calls.")
                        # Keep existing description, title, summary
                    else:
                        # --- LLM Calls --- 
                        if llm_resume and (has_description or has_title or has_summary):
                             logger.info(f"Scene {scene_num} has partial data. Re-running LLM calls.")

                        image_paths = self._find_scene_images(scene, output_dir)
                        logger.debug(f"Calling LLM for description for Scene {scene_num}")
                        description = self._get_llm_description(image_paths, actual_llm_context) # Raises error on failure

                        if description and not description.startswith("Error:"):
                             logger.debug(f"Calling LLM for title/summary for Scene {scene_num}")
                             title, summary = self._generate_title_summary(description) # Raises error on failure
                        else:
                             # Handle case where description generation succeeded but returned an error string 
                             # (less likely with current error handling, but defensive)
                             logger.warning(f"Skipping title/summary generation for Scene {scene_num} due to invalid description: {description}")
                             title = None # Ensure title/summary are None if desc invalid
                             summary = None
                             # Do not raise an error here, let the merge/add logic handle invalid description

                        # --- Update raw_scenes and save JSON snapshot --- 
                        # Update the raw list (for resume capability)
                        raw_scenes[i]['Scene Description'] = description
                        raw_scenes[i]['Scene Title'] = title
                        raw_scenes[i]['Scene Summary'] = summary
                        self._save_scenes_to_json(raw_scenes, scene_data_json_path)
                        # --- End LLM Calls --- 

                    # --- Similarity Check and Merging/Appending --- 
                    scene_to_process = {
                        **scene, 
                        'Scene Description': description,
                        'Scene Title': title,
                        'Scene Summary': summary
                    }

                    if description and not description.startswith("Error:"):
                        should_merge = False
                        if processed_scenes:
                            # Similarity check might raise an error
                            should_merge = self._are_scenes_similar(
                                processed_scenes[-1]['Scene Description'],
                                scene_to_process['Scene Description'],
                                processed_scenes[-1]['Scene Number'],
                                scene_to_process['Scene Number']
                            )

                        if should_merge:
                            logger.debug(f"Merging scene {scene_to_process['Scene Number']} into previous scene {processed_scenes[-1]['Scene Number']}")
                            # Merge might raise an error (e.g., during title/summary regen)
                            processed_scenes[-1] = self._merge_scenes(processed_scenes[-1], scene_to_process)
                        else:
                            logger.debug(f"Adding Scene {scene_to_process['Scene Number']} to processed list.")
                            processed_scenes.append(scene_to_process)
                        
                        scene_processed_successfully = True # Mark success ONLY if added/merged
                    else:
                        logger.warning(f"Skipping addition/merging of Scene {scene_num} due to invalid or missing description.")
                        # scene_processed_successfully remains False

                except Exception as scene_error:
                    # Error occurred during this scene's processing (LLM, merge, etc.)
                    logger.error(f"Error processing Scene {scene_num}: {scene_error}", exc_info=False) # Log concise error here
                    current_error = scene_error # Store the error for this iteration
                    scene_processed_successfully = False
                
                # --- Progressive CSV Saving (happens every iteration) ---
                try:
                    # Write the current state of processed_scenes.
                    # If scene_error occurred, processed_scenes was *not* updated for this scene.
                    self._write_output_csv(processed_scenes, output_csv_path)
                except Exception as csv_error:
                    logger.error(f"Failed to write progressive CSV snapshot after Scene {scene_num}: {csv_error}", exc_info=True)
                    # Prioritize the scene processing error if both occurred
                    if not current_error:
                        current_error = csv_error

                # --- Check if we need to break the loop and store the error --- 
                if current_error:
                    error_to_raise_after_loop = current_error # Store the first error encountered
                    break # Exit the loop immediately

            # --- After the loop --- 
            if error_to_raise_after_loop:
                # An error occurred in the loop, and we broke out
                # Error message was logged when it happened
                logger.critical(f"Processing stopped due to an error during Scene {scene_num}. Check logs above for details.")
                logger.critical("CSV file contains results processed up to the point *before* the error.")
                raise error_to_raise_after_loop # Re-raise the error that caused the break
            else:
                # Loop completed without any errors stored
                logger.info(f"Finished processing successfully. {len(processed_scenes)} final scenes accumulated.")
                # Final CSV write is implicitly handled by the last iteration's write

        except Exception as e:
            # Catch errors from initial setup OR re-raised errors from the loop
            # Avoid double logging if it's the same error we are about to raise
            if error_to_raise_after_loop is None: 
                logger.critical(f"Processing pipeline encountered a critical error: {e}", exc_info=True)
            # Ensure the script terminates with an error status
            raise # Re-raise

    def _write_output_csv(self, scenes: List[Dict[str, Any]], output_csv_path: str):
        """Writes the processed scene data (including descriptions) to a CSV file."""
        if not scenes:
            logger.warning("No scenes to write to output CSV.")
            return
            
        logger.info(f"Writing {len(scenes)} processed scenes to {output_csv_path}")
        
        # Define the header order, including new columns
        # Use fieldnames derived from the structure created in _format_scene_list
        # Add the new fields if not present (though they should be)
        if scenes:
             base_fieldnames = list(scenes[0].keys())
        else: # Define default headers if somehow scenes is empty
             base_fieldnames = [
                 'Scene Number', 'Start Frame', 'Start Timecode', 'Start Time (seconds)',
                 'End Frame', 'End Timecode', 'End Time (seconds)', 'Length (frames)', 
                 'Length (timecode)', 'Length (seconds)', 'Scene Title', 'Scene Summary', 
                 'Scene Description'
             ]

        # Ensure specific order
        ordered_fieldnames = [
             'Scene Number', 'Start Frame', 'Start Timecode', 'Start Time (seconds)',
             'End Frame', 'End Timecode', 'End Time (seconds)', 'Length (frames)', 
             'Length (timecode)', 'Length (seconds)', 'Scene Title', 'Scene Summary', 
             'Scene Description'
        ]
        # Add any extra columns that might exist (less likely now)
        existing_extras = [f for f in base_fieldnames if f not in ordered_fieldnames]
        final_fieldnames = ordered_fieldnames + existing_extras

        try:
            with open(output_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=final_fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(scenes)
            logger.info(f"Successfully wrote output CSV: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to write output CSV {output_csv_path}: {e}")
            raise

    def clear_scene_descriptions(self, scene_data: str):
        """
        Clears the 'Scene Title', 'Scene Summary', and 'Scene Description' fields 
        from a given scene data JSON file.

        Args:
            scene_data: Path to the scene data JSON file (e.g., scenes_data.json).
        """
        logger.info(f"Attempting to clear descriptions from {scene_data}")

        try:
            # Load the scene data
            with open(scene_data, 'r') as f:
                scenes = json.load(f)
            
            if not isinstance(scenes, list):
                 logger.error(f"Invalid format in {scene_data}: Expected a list of scenes.")
                 raise ValueError(f"Invalid format in {scene_data}: Expected a list of scenes.")

            logger.debug(f"Loaded {len(scenes)} scenes from {scene_data}")

            # Clear the fields for each scene
            cleared_count = 0
            for scene in scenes:
                 if isinstance(scene, dict):
                     scene['Scene Title'] = ''
                     scene['Scene Summary'] = ''
                     scene['Scene Description'] = ''
                     cleared_count += 1
                 else:
                      logger.warning(f"Skipping non-dictionary item found in scene list: {type(scene)}")

            logger.info(f"Cleared description fields for {cleared_count} scenes.")

            # Save the modified data back to the same file
            with open(scene_data, 'w') as f:
                json.dump(scenes, f, indent=4)
            
            logger.info(f"Successfully saved cleared scene data back to {scene_data}")

        except FileNotFoundError:
            logger.error(f"Scene data file not found: {scene_data}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {scene_data}: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while clearing scene descriptions in {scene_data}: {e}")
            raise


if __name__ == "__main__":
  fire.Fire(SceneDescriber)
