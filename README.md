This repository contains my implementation of a **Multimodal Retrieval-Augmented Generation (RAG) system for video content interaction**, developed as part of the [Multimodal RAG: Chat with Videos](https://www.deeplearning.ai/short-courses/multimodal-rag-chat-with-videos/) course by DeepLearning.AI in collaboration with Intel. The system enables **intelligent querying and interaction with video content** using **multimodal AI techniques**, including video frame extraction, transcription, multimodal embedding, and retrieval.

## Certificate of Completion

I successfully completed the course! You can view my certificate [here](https://learn.deeplearning.ai/accomplishments/12b98ae0-cf49-4aa6-bb0a-7431fa9232a7?usp=sharing).

## STEP 1: Video preprocessing for ingestion ("1-preprocessing_videos.py")

This step processes 3 different kinds of input videos:

1. Case 1: Video and transcript (commonly in WEBVTT format) are both available. The WEBVTT format consists of sequences of time segments associated with time intervals. In this case, the [OpenCV](https://opencv.org/) library is used to extract video frames using the WEBVTT time stamps. Each text segment gets associated with a central video frame and metadata.

   For each video segment, the following gets extracted:

   - A frame right at the middle of the time frame of the video segment
   - Its metadata including:  
     `extracted_frame_path`: Path to the saved extracted frame  
     `transcript`: Transcript of the extracted frame  
     `video_segment_id`: The order of the video segment from which the frame was extracted  
     `video_path`: Path to the video from which the frame was extracted. This helps retrieving the correct video when there are multiple videos.  
     `mid_time_ms`: Time stamp (in ms) of the extracted frame

2. Case 2: Video with audio is available, but no transcript. In this case, OpenAI's [whisper](https://github.com/openai/whisper) model is used to generate a transcription. Because the whisper model needs an audio file (mp3) as input, the [MoviePy](https://github.com/Zulko/moviepy) library is used to extract audio from the video. The text output from the whisper model is then processed to the WEBVTT format.

3. Case 3: Video with no audio and no transcript (for example silent videos or videos with just background music). In this case, the video gets split into frames and the LLaVA model is used to generate captions from the video frames.

## STEP 2: Ingestion into the LanceDB vector database ("2-vector_store_ingestion.py")

Step 2 uses `BridgeTower/bridgetower-large` as embedding model to ingest the data from Step 1 into a local LanceDB vector database.

The transcripts of frames extracted from a video are usually fragmented and can have incomplete sentences. Such transcripts are often not meaningful and are therefore not helpful for retrieval. In addition, long transcripts that include a lot of information are also not helpful. A simple solution to this issue is to augment such a transcript with the transcripts of "n" neighbouring frames. It is advised to pick an individual "n" (integer number) for each video, so that the updated transcript contains one or two meaningful facts. It is OK to have updated transcripts of neighbouring frames overlapped with each other. Changing the transcriptions which will be ingested into the vector database along with their corresponding frames will affect the performance. It is best to experiment with each video to get the best performance.

## Steps 3a and Step 3b: Run a multimodal RAG system as a chain in LangChain

- The `RunnableParallel` primitive is essentially a dict whose values are runnables (or things that can be coerced to runnables, like functions). It runs all of its values in parallel, and each value is called with the overall input of the `RunnableParallel`. The final return value is a dict with the results of each value under its appropriate key.
- The `RunnablePassthrough` on its own allows to pass inputs unchanged. This is typically used in conjunction with `RunnableParallel` to pass data through to a new key in the map.
- The `RunnableLambda` converts a Python function into a runnable. Wrapping a function in a `RunnableLambda` makes the function usable within either a sync or async context.

The chain where LangChain combines the modules is:

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

### STEP 3a: RAG with LangChain using Python program code ("3a-rag_with_langchain.py")

Step 3a contains several example queries about the ingested videos using Python program code. This is the text-based version of Step 3b. Images are shown separately.

### STEP 3b: Using the Gradio web interface for RAG ("3b-web_interface.py")

Step 3b contains several example queries about the ingested video data via the Gradio web interface. This is the Graphical User Interface (GUI) version of Step 3a. It will take some time to generate the answers, so please be patient.

Follow-up questions will be answered in the context of the currently selected video. To start a new unrelated query, click the "Clear history" button first.

## Required API keys for this example

This example requires API keys from both OpenAI and Prediction Guard.

You need to insert these 2 pieces of information into the `.env.example` file and then rename this file to just `.env` (remove the ".example" ending).

- [Get your OpenAI API key here](https://platform.openai.com/login).
- [Get your free Prediction Guard API key here](https://predictionguard.com/get-started). You will need to fill in the form and ask Prediction Guard to provide you with a free API key for your trial use case.

## Free Prediction Guard API key limitations

Some Prediction Guard trial API keys are limited to 1 or 2 request(s) per second. If your Prediction Guard API key has this limitation, then you must include `time.sleep()` statements, otherwise you might get http status code 429 "Too Many Requests" responses. If your Prediction Guard API key does not have this limitation, then you can remove the existing `time.sleep(1.5)` statements in the `utils.py` file. This will improve the script execution times.

