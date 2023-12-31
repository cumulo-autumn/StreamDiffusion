<script lang="ts">
  import { mediaDevices, mediaStreamActions } from '$lib/mediaStream';
  import Screen from '$lib/icons/screen.svelte';
  import { onMount } from 'svelte';

  let deviceId: string = '';
  $: {
    console.log($mediaDevices);
  }
  $: {
    console.log(deviceId);
  }
  onMount(() => {
    deviceId = $mediaDevices[0].deviceId;
  });
</script>

<div class="flex items-center justify-center text-xs">
  <button
    title="Share your screen"
    class="border-1 my-1 flex cursor-pointer gap-1 rounded-md border-gray-500 border-opacity-50 bg-slate-100 bg-opacity-30 p-1 font-medium text-white"
    on:click={() => mediaStreamActions.startScreenCapture()}
  >
    <span>Share</span>

    <Screen classList={''} />
  </button>
  {#if $mediaDevices}
    <select
      bind:value={deviceId}
      on:change={() => mediaStreamActions.switchCamera(deviceId)}
      id="devices-list"
      class="border-1 block cursor-pointer rounded-md border-gray-800 border-opacity-50 bg-slate-100 bg-opacity-30 p-1 font-medium text-white"
    >
      {#each $mediaDevices as device, i}
        <option value={device.deviceId}>{device.label}</option>
      {/each}
    </select>
  {/if}
</div>
