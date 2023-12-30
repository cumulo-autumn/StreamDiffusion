import { useCallback, useEffect, useRef, useState } from 'react'
import { Box, Container, Flex, Grid, Loader, TextInput } from '@mantine/core'

import styles from './app.module.css'

export const App = () => {
  const [inputPrompt, setInputPrompt] = useState('')
  const [lastPrompt, setLastPrompt] = useState('')
  const [images, setImages] = useState(Array(16).fill('images/white.jpg'))
  const [isLoading, setIsLoading] = useState(false)
  const abortControllers = useRef<AbortController[]>([])

  const calculateEditDistance = (a: string, b: string) => {
    if (a.length === 0) return b.length
    if (b.length === 0) return a.length

    const matrix = []

    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i]
    }
    for (let i = 0; i <= a.length; i++) {
      matrix[0]![i] = i
    }

    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          //@ts-ignore
          matrix[i][j] = matrix[i - 1][j - 1]
        } else {
          //@ts-ignore
          matrix[i][j] = Math.min(
            //@ts-ignore
            matrix[i - 1][j - 1] + 1,
            //@ts-ignore
            Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1),
          )
        }
      }
    }

    //@ts-ignore
    return matrix[b.length][a.length]
  }

  const fetchImage = useCallback(
    async (index: number): Promise<void> => {
      abortControllers.current[index]?.abort()
      abortControllers.current[index] = new AbortController()
      const signal = abortControllers.current[index]?.signal

      setIsLoading(true)
      try {
        const response = await fetch('api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: inputPrompt }),
          signal,
        })
        const data = await response.json()
        const imageUrl = `data:image/jpeg;base64,${data.base64_image}`

        setImages((prevImages) => {
          const newImages = [...prevImages]
          newImages[index] = imageUrl
          return newImages
        })
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          console.error('Error fetching image:', error)
        }
      } finally {
        setIsLoading(false)
      }
    },
    [inputPrompt],
  )

  const handlePromptChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const newPrompt = event.target.value
    setInputPrompt(newPrompt)
    const editDistance = calculateEditDistance(lastPrompt, newPrompt)

    if (editDistance && editDistance >= 4) {
      setLastPrompt(newPrompt)
      // setImages(Array(16).fill('images/white.jpg'));
      for (let i = 0; i < 16; i++) {
        fetchImage(i)
      }
    }
  }

  useEffect(() => {
    return () => {
      abortControllers.current.forEach((controller) => controller.abort())
    }
  }, [])

  return (
    <Box bg="#282c34" mih="100vh" w="100vw" p="lg">
      <Container className={styles.container}>
        <Flex direction="column" justify="center" align="center">
          <Grid w="100%" justify="center" align="center">
            {images.map((image, index) => (
              <Grid.Col
                span={3}
                key={index}
                style={{
                  textAlign: 'center',
                }}
              >
                <img
                  src={image}
                  alt={`Generated ${index}`}
                  style={{
                    maxWidth: '100%',
                    maxHeight: '150px',
                    borderRadius: '10px',
                  }}
                />
              </Grid.Col>
            ))}
          </Grid>
          <TextInput w="100%" size="lg" placeholder="Enter a prompt" value={inputPrompt} onChange={handlePromptChange} />
          {isLoading && <Loader />}
        </Flex>
      </Container>
    </Box>
  )
}
