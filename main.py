from ml4a import image
from ml4a import audio
from mechanicalduck import cli
from mechanicalduck.models import gpt3
from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block


eden_block = BaseBlock(max_gpu_mem = 1)

my_args = {
    'question': 'What is the nature of consciousness?'
}
@eden_block.run(args = my_args, progress = True)
def do_something(config): 

    question = config['question']
    device = config['__gpu__']

    prompt = '''
    Question: What language is the most useful in Europe and Russia (except English and Russian)? Why?
    Answer: The answer is German. German is widely studied in all the eastern European states bordering Germany and Austria - over 50% of students. And what’s more, students generally are actually interested in it and study it to a reasonably conversant level.


    Question: What are some fascinating examples of ancient or medieval technology?
    Answer: Ancient civilizations did their most impressive work with water. Hydraulic engineering is where the Egyptians, the Persians, the Greeks, the Romans and others devoted the most resources by far and they had some impressive achievements to display.


    Question: How would you explain the essence of Bhagavad Gita?
    Answer: Here is the backdrop. A great warrior who is fighting against injustice is suddenly overcome by sorrow. He had to fight a war against everyone he cared for - his cousins, teacher, uncles, classmates.. Overtaken by emotions, he attempts to give up the war.

    Question: What is the nature of consciousness?
    Answer:'''

    face_file = 'stylegan_temp.mp4'
    speech_file = 'speech_temp.wav'
    output_file = 'final.mp4'


    response = gpt3.complete(prompt, 
        stops=['\n', 'Answer:', 'Question:'], 
        max_tokens=100, 
        temperature=0.9, 
        engine='davinci',
        max_completions=1)

    print(response)

    cli.tacotron2(speech_file, 
        response, 
        as_subprocess=True)

    w, sr = audio.load(speech_file)
    duration = len(w)/sr

    print("=> finished Tacotron, %0.2f sec"%duration)

    cli.stylegan(face_file, 'ffhq', 
        duration_sec=duration,
        smoothing_sec=2.0, 
        truncation=1.0, 
        as_subprocess=True)

    print("=> finished StyleGAN video")

    cli.wav2lip(output_file,
        face_file, 
        speech_file, 
        as_subprocess=True)

    print("=> finished Wav2Lip video")

    return {
        'question': config['question'],  ## returning text
        'response': response
    }


host_block(
    eden_block,
    port = 5656,
    max_num_workers = 2,
    redis_port = 6379,
    exclude_gpu_ids = [],
    logfile = 'log.txt',
    log_level = 'error'
)