import '@mantine/core/styles.css'

import React from 'react'
import ReactDOM from 'react-dom/client'
import { MantineProvider } from '@mantine/core'

import { App } from './app'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <MantineProvider>
      <App />
    </MantineProvider>
  </React.StrictMode>,
)
