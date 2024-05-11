import React, { useState } from "react";
import Switch from "react-switch";
import Threads from "./Threads";
import darkmode from "../Assets/darkmode.svg";
const Home = () => {
  const [checked, setchecked] = useState(false);
  const [threads, setthreads] = useState(false);

  const handleClick = () => {
    if (checked) {
      setchecked(false);
    } else {
      setchecked(true);
    }
  };

  return (
    <div className="bg-[#F7BDA2] w-[100vw] h-[100vh]">
      {threads ? (
        <Threads />
      ) : (
        <div>
          {/* <div className="flex justify-end p-4">
            <img src={darkmode} className="px-4" />
            <Switch
              onChange={handleClick}
              checked={checked}
              onColor={"#FFE600"}
              handleDiameter={30}
              uncheckedIcon={false}
              checkedIcon={false}
              boxShadow="0px 1px 5px rgba(0, 0, 0, 0.6)"
              activeBoxShadow="0px 0px 1px 10px rgba(0, 0, 0, 0.2)"
              height={20}
              width={48}
              className="react-switch"
              id="material-switch"
            />
          </div> */}
          <div className="mt-[22vh] flex flex-col mx-10 items-center justify-center">
            <button
              className="bg-[#48161F]  rounded-2xl p-2 w-80 m-3 text-lg text-white font-serif"
              onClick={() => setthreads(true)}
            >
              Seek legal clarity? Chat with me
            </button>
            {/* <img src={Logo} alt="Logo" className="w-[14vw] py-6  mx-auto" /> */}
            {/* <img src="C:/Users/hp/OneDrive/Pictures/Screenshots/Logo.png" alt="Logo" width="100" height="100" style={{ backgroundColor: "transparent" }} /> */}

          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
