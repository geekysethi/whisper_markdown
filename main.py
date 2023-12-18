import tkinter as tk
import threading
import time

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import sounddevice as sd
import numpy as np
import sys

def model_inference(speech):

	speech = np.asarray(speech).astype(np.float32)
	speech = speech.astype(np.float32) / np.iinfo(np.int16).max

	model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
	input_features = processor(speech, return_tensors="pt", sampling_rate=16000,).input_features
	predicted_ids = model.generate(input_features)
	transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True, normalize = False)

	return transcription


def recording():
	global recording_active
	global recording_data

	sample_rate = 16000  # Adjust as needed
	with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype=np.int16):
		print("Recording... Press the 'Stop Recording' button to stop.")
		while recording_active:

			print("RECORDING...")
			time.sleep(1)
	transcription = model_inference(recording_data)
	
	final_text = " ".join(transcription)
	final_text = final_text + '\n'
	with open(markdown_file_path, 'a', encoding='utf-8') as file:
		file.write(final_text)

	# sample_rate = 16000 


def callback(indata, frames, time, status):
	global recording_data
	if status:
		print(status, file=sys.stderr)
	recording_data.extend(indata[:,0])




def toggle_recording():
	global recording_active
	global recording_data
	recording_active = not recording_active

	if recording_active:
		recording_data = []

		recording_thread = threading.Thread(target=recording)
		recording_thread.start()
		button.config(text="Stop", command=toggle_recording)

	

	else:		
		button.config(text="Start", command=toggle_recording)


if __name__ == "__main__":


	markdown_file_path = "output.md"
	with open(markdown_file_path, 'w', encoding='utf-8') as file:
		file.write("")

	processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
	model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")

	recording_active = False
	recording_data = []
	root = tk.Tk()
	root.geometry("400x400")
	button = tk.Button(root, text='Start', command=toggle_recording, width=5, height=2)
	button.pack(pady=(150, 0))
	button.pack()
	root.mainloop()







