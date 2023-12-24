<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  export let value = 299792458;
  export let params: FieldProps;

  onMount(() => {
    value = Number(params?.default ?? '');
  });
  function randomize() {
    value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-3">
  <label class="text-sm font-medium" for="seed">Seed</label>
  <input
    bind:value
    type="number"
    id="seed"
    name="seed"
    class="col-span-2 rounded-md border border-gray-700 p-2 text-right font-light dark:text-black"
  />
  <Button on:click={randomize}>Rand</Button>
</div>
