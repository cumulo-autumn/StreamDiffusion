<script lang="ts">
  export let classList: string = 'p-2';
  export let disabled: boolean = false;
  export let title: string = '';
</script>

<button class="button {classList}" on:click {disabled} {title}>
  <slot />
</button>

<style lang="postcss" scoped>
  .button {
    @apply rounded bg-gray-700 font-normal text-white hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-300 dark:disabled:bg-gray-700 dark:disabled:text-black;
  }
</style>
