import asyncio
import websockets

async def handler(websocket):
    """Handle incoming websocket connections and messages"""
    client_address = websocket.remote_address
    print(f"Client connected from {client_address}")

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"Received {len(message)} bytes from {client_address}")
                # Send acknowledgment response
                await websocket.send(b'\x00' * 8)
            else:
                print(f"Warning: Received non-binary message from {client_address}: {type(message)}")

    except websockets.exceptions.ConnectionClosed:
        print(f"Client {client_address} disconnected")
    except Exception as e:
        print(f"Error handling client {client_address}: {e}")

async def main():
    """Main server function"""
    host = "localhost"
    port = 8765

    print(f"Starting websocket server on {host}:{port}")

    # Start the server
    async with websockets.serve(handler, host, port):
        print("Websocket server is running...")
        # Keep the server running
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server shutdown requested")
    except Exception as e:
        print(f"Server error: {e}")

