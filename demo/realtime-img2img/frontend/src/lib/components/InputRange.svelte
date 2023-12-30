<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = 8.0;
  export let params: FieldProps;
  onMount(() => {
    value = Number(params?.default) ?? 8.0;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-3">
  <label class="text-sm font-medium" for={params.id}>{params?.title}</label>
  <input
    class="col-span-2 h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-300 dark:bg-gray-500"
    bind:value
    type="range"
    id={params.id}
    name={params.id}
    min={params?.min}
    max={params?.max}
    step={params?.step ?? 1}
  />
  <input
    type="number"
    step={params?.step ?? 1}
    bind:value
    class="rounded-md border px-1 py-1 text-center text-xs font-bold dark:text-black"
  />
</div>
<!-- 
<style lang="postcss" scoped>
  input[type='range']::-webkit-slider-runnable-track {
    @apply h-2 cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-webkit-slider-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-moz-range-track {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-moz-range-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-ms-track {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
  input[type='range']::-ms-thumb {
    @apply cursor-pointer rounded-lg dark:bg-gray-50;
  }
</style> -->
