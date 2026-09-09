import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { SWRConfig } from "swr";

import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <SWRConfig
      value={{
        refreshInterval: 0,
        revalidateOnFocus: true,
        fetcher: async (path: string) => {
          const r = await fetch(path);
          if (!r.ok) {
            const text = await r.text();
            const err = new Error(`API ${r.status}: ${text || r.statusText}`);
            (err as any).status = r.status;
            throw err;
          }
          return r.json();
        },
      }}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </SWRConfig>
  </React.StrictMode>
);
