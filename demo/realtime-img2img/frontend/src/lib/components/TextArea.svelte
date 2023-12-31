<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value: string;
  export let params: FieldProps;
  onMount(() => {
    value = String(params?.default ?? '');
  });
</script>

<div class="">
  <label class="text-sm font-medium" for={params?.title}>
    {params?.title}
  </label>
  <div class="text-normal flex items-center rounded-md border border-gray-700">
    <textarea
      class="mx-1 w-full px-3 py-2 font-light outline-none dark:text-black"
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value
    ></textarea>
  </div>
</div>
