import React, { useState } from "react";

import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import Mic from "../Assets/mic.svg";
import OnMic from "../Assets/micon.svg";
var lang="en"
const Dictaphone = ({ input, setinput }) => {
  const [mic, setMic] = useState(false);
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();

  if (!browserSupportsSpeechRecognition) {
    return alert("Browser doesn't support speech recognition");
  }
  const onMicStart = () => {
    setMic(!mic);
    mic
      ? SpeechRecognition.startListening({language:lang})
      : SpeechRecognition.stopListening();
    console.log(transcript);
    if(transcript=="Tamil" || transcript == "தமிழ்" || transcript == "तमिल" ){
      lang="ta-IN"
    }
    if (transcript=="இங்கிலீஷ்" || transcript =="English" || transcript == "अंग्रेज़ी"){
      lang="en"
    }
    if(transcript == "Hindi" || transcript == "ஹிந்தி" || transcript =="हिंदी"){
      lang = "hi-IN"
    }
    setinput(transcript);
  };

  return (
    <div>
      <img
        src={listening ? OnMic : Mic}
        alt="Mic"
        className="w-[4.5vw] h-[5vh] mt-2 px-0 cursor-pointer transition-all user-select-none"
        onClick={onMicStart}
      />
    </div>
  );
};
export default Dictaphone;
