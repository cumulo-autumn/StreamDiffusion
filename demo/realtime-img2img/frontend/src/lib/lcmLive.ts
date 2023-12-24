import { writable } from 'svelte/store';


export enum LCMLiveStatus {
    CONNECTED = "connected",
    DISCONNECTED = "disconnected",
    WAIT = "wait",
    SEND_FRAME = "send_frame",
}

const initStatus: LCMLiveStatus = LCMLiveStatus.DISCONNECTED;

export const lcmLiveStatus = writable<LCMLiveStatus>(initStatus);
export const streamId = writable<string | null>(null);

let websocket: WebSocket | null = null;
export const lcmLiveActions = {
    async start(getSreamdata: () => any[]) {
        return new Promise((resolve, reject) => {

            try {
                const websocketURL = `${window.location.protocol === "https:" ? "wss" : "ws"
                    }:${window.location.host}/api/ws`;

                websocket = new WebSocket(websocketURL);
                websocket.onopen = () => {
                    console.log("Connected to websocket");
                };
                websocket.onclose = () => {
                    lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                    console.log("Disconnected from websocket");
                };
                websocket.onerror = (err) => {
                    console.error(err);
                };
                websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    switch (data.status) {
                        case "connected":
                            const userId = data.userId;
                            lcmLiveStatus.set(LCMLiveStatus.CONNECTED);
                            streamId.set(userId);
                            resolve(userId);
                            break;
                        case "send_frame":
                            lcmLiveStatus.set(LCMLiveStatus.SEND_FRAME);
                            const streamData = getSreamdata();
                            websocket?.send(JSON.stringify({ status: "next_frame" }));
                            for (const d of streamData) {
                                this.send(d);
                            }
                            break;
                        case "wait":
                            lcmLiveStatus.set(LCMLiveStatus.WAIT);
                            break;
                        case "timeout":
                            console.log("timeout");
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject("timeout");
                        case "error":
                            console.log(data.message);
                            lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                            streamId.set(null);
                            reject(data.message);
                    }
                };

            } catch (err) {
                console.error(err);
                lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
                streamId.set(null);
                reject(err);
            }
        });
    },
    send(data: Blob | { [key: string]: any }) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            if (data instanceof Blob) {
                websocket.send(data);
            } else {
                websocket.send(JSON.stringify(data));
            }
        } else {
            console.log("WebSocket not connected");
        }
    },
    async stop() {

        if (websocket) {
            websocket.close();
        }
        websocket = null;
        lcmLiveStatus.set(LCMLiveStatus.DISCONNECTED);
        streamId.set(null);
    },
};