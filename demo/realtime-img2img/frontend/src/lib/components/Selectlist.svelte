<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = '';
  export let params: FieldProps;
  onMount(() => {
    value = String(params?.default);
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center justify-items-start gap-3">
  <label for="model-list" class="text-sm font-medium">{params?.title} </label>
  {#if params?.values}
    <select
      bind:value
      id="model-list"
      class="cursor-pointer rounded-md border-2 border-gray-500 p-2 font-light dark:text-black"
    >
      {#each params.values as model, i}
        <option value={model} selected={i === 0}>{model}</option>
      {/each}
    </select>
  {/if}
</div>
