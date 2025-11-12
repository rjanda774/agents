"""LLM-driven orchestration for the MCP video tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from .mcp_tools import (
    get_video_metadata,
    compress_video,
    extract_audio,
    convert_format,
    resize_video,
    extract_thumbnail,
    trim_video,
)

TOOL_MAP = {
    "get_video_metadata": get_video_metadata,
    "compress_video": compress_video,
    "extract_audio": extract_audio,
    "convert_format": convert_format,
    "resize_video": resize_video,
    "extract_thumbnail": extract_thumbnail,
    "trim_video": trim_video,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_video_metadata",
            "description": "Get detailed metadata about a video file including duration, size, resolution, codecs",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    }
                },
                "required": ["video_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_video",
            "description": "Compress a video to reduce file size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Compression quality",
                    },
                },
                "required": ["video_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_audio",
            "description": "Extract audio from a video file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["mp3", "wav", "aac", "flac"],
                        "description": "Target audio format",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Audio quality",
                    },
                },
                "required": ["video_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_format",
            "description": "Convert video to a different container/codec.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["mp4", "webm", "avi", "mov", "mkv", "gif"],
                        "description": "Desired format",
                    },
                },
                "required": ["video_path", "output_format"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resize_video",
            "description": "Resize video to a preset resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": ["480p", "720p", "1080p", "1440p", "4k"],
                        "description": "Target resolution",
                    },
                },
                "required": ["video_path", "resolution"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_thumbnail",
            "description": "Extract a thumbnail image from a video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "timestamp": {
                        "type": "number",
                        "description": "Timestamp in seconds",
                    },
                },
                "required": ["video_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trim_video",
            "description": "Extract a segment from the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {
                        "type": "string",
                        "description": "Full path to the video file",
                    },
                    "start_time": {
                        "type": "number",
                        "description": "Start time in seconds",
                    },
                    "end_time": {
                        "type": "number",
                        "description": "End time in seconds",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds",
                    },
                },
                "required": ["video_path"],
            },
        },
    },
]


async def execute_tool(tool_name: str, arguments: Dict[str, Any]):
    tool = TOOL_MAP.get(tool_name)
    if not tool:
        return {"error": f"Unknown tool: {tool_name}"}
    return await tool(**arguments)


async def process_request(
    video_path: str,
    user_request: str,
    llm_client,
    model: str,
) -> Tuple[str, Optional[List[str]]]:
    if not video_path:
        return "Please upload a video first!", None

    video_info = await get_video_metadata(video_path)
    system_prompt = f"""You are an autonomous video processing agent. The user has uploaded a video:

CURRENT VIDEO STATE:
- File: {Path(video_path).name}
- Duration: {video_info.get('duration_formatted', 'unknown')}
- Size: {video_info.get('size_mb', 'unknown')}MB
- Resolution: {video_info.get('width', '?')}x{video_info.get('height', '?')}
- Format: {video_info.get('format', 'unknown')}

USER GOAL: {user_request}

IMPORTANT RULES:
1. ALWAYS use this exact video_path: {video_path}
2. You can call MULTIPLE tools to achieve the goal
3. After each tool, evaluate if you need to do more
4. For complex requests (like "optimize for Instagram"), break it into steps:
   - First, understand requirements (Instagram: square, <100MB, etc.)
   - Then execute operations in order (resize → compress → verify)
5. When you update the video_path (after resize/compress/convert), use the NEW path for next operations
6. Only stop when the user's goal is fully achieved

AVAILABLE TOOLS:
- get_video_metadata: Check current video properties
- compress_video: Reduce file size (quality: low/medium/high)
- extract_audio: Get audio (formats: mp3/wav/aac/flac)
- convert_format: Change format (mp4/webm/avi/mov/mkv/gif)
- resize_video: Change resolution (480p/720p/1080p/1440p/4k)
- extract_thumbnail: Get preview image (specify timestamp)
- trim_video: Extract a segment/clip (use start_time + duration OR start_time + end_time)

PLATFORM REQUIREMENTS (use these when user mentions platforms):
- Instagram: Square (1080x1080), <100MB, MP4
- YouTube: 1080p or higher, MP4/WebM
- Twitter: <512MB, MP4/MOV
- Email: <25MB, compressed, 720p or lower
- WhatsApp: <16MB, MP4, 720p

Think step-by-step and execute all necessary operations to achieve the user's goal."""

    max_iterations = 5
    iteration = 0
    current_video_path = video_path
    final_output_path = None
    all_output_files: List[str] = []
    operations_log: List[str] = []

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ]

        while iteration < max_iterations:
            iteration += 1
            response = llm_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    if "video_path" in arguments:
                        arg_path = arguments["video_path"]
                        if not os.path.exists(arg_path):
                            if (
                                os.path.basename(current_video_path) == arg_path
                                or arg_path == video_path
                            ):
                                arguments["video_path"] = current_video_path

                    result = await execute_tool(tool_name, arguments)
                    operations_log.append(f"✓ {tool_name}")

                    if "error" in result:
                        messages.append(
                            {
                                "role": "tool",
                                "content": json.dumps({"error": result["error"]}),
                                "tool_call_id": tool_call.id,
                            }
                        )
                        return (
                            f"❌ Error in step {iteration}: {result['error']}\n\nCompleted: {', '.join(operations_log)}",
                            final_output_path,
                        )

                    output_path = result.get("output_path")
                    if output_path:
                        all_output_files.append(output_path)
                        final_output_path = output_path
                        current_video_path = output_path

                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id,
                        }
                    )
            else:
                final_message = message.content
                steps_summary = (
                    f"\n\n🔧 Steps executed: {' → '.join(operations_log)}"
                    if len(operations_log) > 1
                    else ""
                )
                return f"✅ {final_message}{steps_summary}", (
                    all_output_files if all_output_files else None
                )

        return (
            f"⚠️ Completed {iteration} steps but may not be fully done.\n\n🔧 Steps: {' → '.join(operations_log)}",
            all_output_files if all_output_files else None,
        )

    except Exception as exc:  # pragma: no cover - defensive logging
        return (
            f"❌ Error processing request: {exc}\n\nCompleted steps: {', '.join(operations_log)}",
            all_output_files if all_output_files else None,
        )


__all__ = ["TOOLS", "process_request"]
