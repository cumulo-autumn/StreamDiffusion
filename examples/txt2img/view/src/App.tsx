import React, { useCallback, useEffect, useState } from "react";
import { TextField, Grid, Paper } from "@mui/material";

function App() {
  const [inputPrompt, setInputPrompt] = useState("");
  const [images, setImages] = useState(Array(9).fill("images/white.jpg"));

  const fetchImages = useCallback(async () => {
    try {
      const response = await fetch("http://127.0.0.1:9090/predict", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: inputPrompt })
      });
      const data = await response.json();
      const imageUrls = data.base64_images.map((base64: string) => `data:image/jpeg;base64,${base64}`);
      setImages(imageUrls);
    } catch (error) {
      console.error("Error fetching images:", error);
    }
  }, [inputPrompt]);

  const handlePromptChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputPrompt(event.target.value);
    fetchImages();
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
      <Grid container spacing={2}>
        {images.map((image, index) => (
          <Grid item xs={4} key={index}>
            <Paper style={{ padding: "10px", textAlign: "center" }}>
              <img src={image} alt={`Generated ${index}`} style={{ maxWidth: "100%", maxHeight: "200px", borderRadius: "10px" }} />
            </Paper>
          </Grid>
        ))}
      </Grid>
        <TextField
          variant="outlined"
          value={inputPrompt}
          onChange={handlePromptChange}
          style={{ marginBottom: "20px", marginTop: "20px", width: "640px", color: "#ffffff", borderColor: "#ffffff", borderRadius: "10px", backgroundColor: "#ffffff" }}
          placeholder="Enter a prompt"
        />
      </div>
    </div>
  );
}

export default App;
