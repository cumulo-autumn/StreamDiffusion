import React, { useCallback, useState } from "react";
import { TextField, Grid } from "@mui/material";

function App() {
  const [inputPrompt, setInputPrompt] = useState("");
  const [lastPrompt, setLastPrompt] = useState("");
  const [images, setImages] = useState(Array(16).fill("images/white.jpg"));

  const calculateEditDistance = (a: string, b: string) => {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    for (let i = 0; i <= a.length; i++) {
      matrix[0][i] = i;
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
          );
        }
      }
    }

    return matrix[b.length][a.length];
  };

  const fetchImage = useCallback(
    async (index: number) => {
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: inputPrompt }),
        });
        const data = await response.json();
        const imageUrl = `data:image/jpeg;base64,${data.base64_image}`;

        setImages((prevImages) => {
          const newImages = [...prevImages];
          newImages[index] = imageUrl;
          return newImages;
        });
      } catch (error) {
        console.error("Error fetching image:", error);
      }
    },
    [inputPrompt]
  );

  const handlePromptChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputPrompt(event.target.value);
    const newPrompt = event.target.value;
    const editDistance = calculateEditDistance(lastPrompt, newPrompt);

    if (editDistance >= 2) {
      setInputPrompt(newPrompt);
      setLastPrompt(newPrompt);
      for (let i = 0; i < 16; i++) {
        fetchImage(i);
      }
    }
  };

  return (
    <div
      className="App"
      style={{
        backgroundColor: "#282c34",
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        margin: "0",
        color: "#ffffff",
        padding: "20px",
      }}
    >
      <div
        style={{
          backgroundColor: "#282c34",
          alignItems: "center",
          justifyContent: "center",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Grid
          container
          spacing={1}
          style={{ maxWidth: "50%", maxHeight: "70%" }}
        >
          {images.map((image, index) => (
            <Grid item xs={3} key={index}>
              <img
                src={image}
                alt={`Generated ${index}`}
                style={{
                  maxWidth: "100%",
                  maxHeight: "150px",
                  borderRadius: "10px",
                }}
              />
            </Grid>
          ))}
        </Grid>
        <TextField
          variant="outlined"
          value={inputPrompt}
          onChange={handlePromptChange}
          style={{
            marginBottom: "20px",
            marginTop: "20px",
            width: "640px",
            color: "#ffffff",
            borderColor: "#ffffff",
            borderRadius: "10px",
            backgroundColor: "#ffffff",
          }}
          placeholder="Enter a prompt"
        />
      </div>
    </div>
  );
}

export default App;
