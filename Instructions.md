# Multimodal Retrieval-Augmented Generation (RAG) Chat with Videos

This project demonstrates a question-answering system using multimodal data, allowing users to ask questions and retrieve information from a collection of ingested videos. The system utilizes the API of [Prediction Guard](https://predictionguard.com/), but it can be adapted to use other model hosting providers.

## Overview
The process consists of multiple steps, starting with video preprocessing, followed by ingestion into a vector database, and finally, retrieval and response generation through a multimodal RAG system. Users can interact with the system via:
- A Python program (STEP_3a)
- A web interface powered by [Gradio](https://www.gradio.app/) (STEP_3b)

The Gradio interface is accessible at: `http://localhost:9999`

## System Components
This system uses the following technologies:
- **BridgeTower embedding model**: Embeds text and images into a common semantic space.
- **LanceDB**: A local vector database for storing extracted video data.
- **Large Vision-Language Model (LVLM)**: A model combining visual and textual understanding for retrieval tasks.
- **LLaVA (Large Language and Vision Assistant)**: A powerful multimodal model integrating CLIP’s vision capabilities with LLaMA’s textual understanding.

## Steps
### STEP 1: Video Preprocessing (`STEP_1-preprocessing_videos.py`)
This step processes three types of video inputs:

1. **Videos with transcripts (WEBVTT format)**
   - Extracts frames from the video based on the transcript’s timestamps.
   - Associates each extracted frame with metadata including:
     - Frame path
     - Transcript
     - Video segment ID
     - Video path
     - Timestamp (in milliseconds)

2. **Videos with audio but no transcript**
   - Uses [OpenAI's Whisper](https://github.com/openai/whisper) model for transcription.
   - Converts the transcript into WEBVTT format for further processing.

3. **Videos with no audio and no transcript**
   - Splits the video into frames.
   - Uses the LLaVA model to generate captions from extracted frames.

### STEP 2: Data Ingestion into LanceDB (`STEP_2-vector_store_ingestion.py`)
This step embeds the processed video data into a local LanceDB vector database using `BridgeTower/bridgetower-large`. The system optimizes transcript retrieval by:
- Augmenting fragmented transcripts with neighboring frames’ transcripts to ensure completeness.
- Experimenting with different augmentation strategies to maximize retrieval effectiveness.

### STEP 3: Running the Multimodal RAG System

The retrieval system is implemented as a LangChain processing chain:

```
mm_rag_chain = (
    RunnableParallel({
        "retrieved_results": retriever_module ,
        "user_query": RunnablePassthrough()
    })
    | prompt_processing_module
    | lvlm_inference_module
)
```

#### STEP 3a: Querying via Python (`STEP_3a-rag_with_langchain.py`)
- Runs queries programmatically in Python.
- Returns text-based responses with relevant retrieved frames and metadata.

#### STEP 3b: Querying via Gradio Web Interface (`STEP_3b-web_interface.py`)
- Provides a user-friendly GUI.
- Displays retrieved images along with text responses.
- Allows follow-up questions within the context of a selected video.
- Includes a "Clear history" button to reset the query context.

## API Key Requirements
To use this system, you need API keys for:
1. **OpenAI** ([Get your key here](https://platform.openai.com/login))
2. **Prediction Guard** ([Get your free key here](https://predictionguard.com/get-started))

Add these keys to a `.env` file as follows:
```
OPENAI_API_KEY=your_openai_api_key
PREDICTION_GUARD_API_KEY=your_prediction_guard_api_key
```

### API Rate Limits
Some Prediction Guard trial API keys may have a rate limit (e.g., 1–2 requests per second). If your key has this limitation, you need to include `time.sleep()` statements in your scripts to avoid `429 Too Many Requests` errors. If your key has no such limitation, you can remove the existing `time.sleep(1.5)` statements in `utils.py` to speed up execution.

## FFmpeg Requirement
The system requires **FFmpeg**, a tool for processing audio and video. Install it from the [official FFmpeg website](https://www.ffmpeg.org/).

For Windows users:
- Download `ffmpeg.exe` and place it in the project directory.

For Linux users:
- Install a static build if package installation fails ([get it here](https://johnvansickle.com/ffmpeg/)).

If FFmpeg is missing, you may encounter errors like:
```
FileNotFoundError: The system cannot find the file specified
```
This means the FFmpeg executable is not found, not that the input video is missing.

---

This project provides a powerful framework for building multimodal retrieval-augmented generation systems with videos. Experiment with transcript augmentation, embedding strategies, and retrieval parameters to optimize performance for your specific use case.

